"""
RWKV Model

An RWKV-4-style recurrent language model that is a drop-in counterpart to
AGI2Model. Instead of attending over a growing context window, each layer
carries a fixed-size state forward one token at a time: time-mixing (the
attention counterpart) computes an exponentially-decayed weighted average of
past values, and channel-mixing (the FFN counterpart) is gated by a one-token
lookback.

The model exposes the same interface as AGI2Model (forward, forward_hidden,
_run_transformer, token_embeddings, get_num_params, config) so the existing
training and generation pipeline works unchanged. It reuses AGI2Config;
n_head is unused and no position embeddings are needed — order comes from
the recurrence itself.

Reference: "RWKV: Reinventing RNNs for the Transformer Era" (Peng et al., 2023).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AGI2Config
from .embeddings import TokenEmbeddings


def _token_shift(x: torch.Tensor) -> torch.Tensor:
    """Shift the sequence right by one step (the previous token's features).

    Args:
        x: Tensor of shape (batch_size, seq_len, d_model)

    Returns:
        Tensor of the same shape where position t holds x[:, t-1] (zeros at t=0).
    """
    return F.pad(x, (0, 0, 1, -1))


class RWKVTimeMix(nn.Module):
    """
    Time-mixing block: RWKV's counterpart to self-attention.

    Keeps a per-channel exponentially-decaying average of past values,
    weighted by learned keys, and gates the result with a receptance signal.

    Args:
        d_model: Dimension of the model
        layer_id: Index of this layer (used for initialization schedules)
        n_layer: Total number of layers
    """

    def __init__(self, d_model: int, layer_id: int, n_layer: int):
        super().__init__()
        self.d_model = d_model

        ratio_0_to_1 = layer_id / max(n_layer - 1, 1)
        ratio_1_to_almost0 = 1.0 - layer_id / n_layer
        channel = torch.arange(d_model, dtype=torch.float32) / max(d_model - 1, 1)

        # Per-channel decay speed: slow-decaying channels remember further back.
        decay_speed = -5.0 + 8.0 * channel ** (0.7 + 1.3 * ratio_0_to_1)
        self.time_decay = nn.Parameter(decay_speed)

        # Bonus applied to the current token so it is never averaged away.
        zigzag = 0.5 * (torch.arange(1, d_model + 1, dtype=torch.float32) % 3 - 1)
        self.time_first = nn.Parameter(torch.full((d_model,), -0.6) + zigzag)

        # Interpolation factors between the current and previous token.
        mix = (torch.arange(d_model, dtype=torch.float32) / d_model).view(1, 1, -1)
        self.time_mix_k = nn.Parameter(mix**ratio_1_to_almost0)
        self.time_mix_v = nn.Parameter(mix**ratio_1_to_almost0 + 0.3 * ratio_0_to_1)
        self.time_mix_r = nn.Parameter(mix ** (0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def _wkv(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Numerically stable WKV recurrence over the time dimension.

        Runs in float32 regardless of input dtype; exponentials are kept
        stable by tracking a running maximum exponent per channel.

        Args:
            k: Keys of shape (batch_size, seq_len, d_model)
            v: Values of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = k.size()
        dtype = k.dtype
        k = k.float()
        v = v.float()
        w = -torch.exp(self.time_decay.float())
        u = self.time_first.float()

        aa = k.new_zeros(batch_size, d_model)  # weighted sum of values
        bb = k.new_zeros(batch_size, d_model)  # sum of weights
        pp = k.new_full((batch_size, d_model), -1e38)  # running max exponent
        outputs: List[torch.Tensor] = []

        for t in range(seq_len):
            kt = k[:, t]
            vt = v[:, t]

            ww = u + kt
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            outputs.append((e1 * aa + e2 * vt) / (e1 * bb + e2))

            ww = pp + w
            p = torch.maximum(ww, kt)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kt - p)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = p

        return torch.stack(outputs, dim=1).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through time-mixing.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        prev = _token_shift(x)
        xk = x * self.time_mix_k + prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + prev * (1 - self.time_mix_r)

        r = torch.sigmoid(self.receptance(xr))
        wkv = self._wkv(self.key(xk), self.value(xv))
        return self.output(r * wkv)


class RWKVChannelMix(nn.Module):
    """
    Channel-mixing block: RWKV's counterpart to the feed-forward network.

    A squared-ReLU MLP whose output is gated by a receptance signal computed
    from the current and previous token.

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the hidden layer
        layer_id: Index of this layer (used for initialization schedules)
        n_layer: Total number of layers
    """

    def __init__(self, d_model: int, d_ff: int, layer_id: int, n_layer: int):
        super().__init__()
        ratio_1_to_almost0 = 1.0 - layer_id / n_layer
        mix = (torch.arange(d_model, dtype=torch.float32) / d_model).view(1, 1, -1)
        self.time_mix_k = nn.Parameter(mix**ratio_1_to_almost0)
        self.time_mix_r = nn.Parameter(mix**ratio_1_to_almost0)

        self.key = nn.Linear(d_model, d_ff, bias=False)
        self.value = nn.Linear(d_ff, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through channel-mixing.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        prev = _token_shift(x)
        xk = x * self.time_mix_k + prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + prev * (1 - self.time_mix_r)

        k = torch.square(torch.relu(self.key(xk)))
        r = torch.sigmoid(self.receptance(xr))
        return r * self.value(k)


class RWKVBlock(nn.Module):
    """
    RWKV block combining time-mixing and channel-mixing with layer norms.

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the channel-mixing hidden layer
        layer_id: Index of this layer
        n_layer: Total number of layers
        dropout: Dropout rate for residual connections
        layer_norm_epsilon: Epsilon for layer normalization
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        layer_id: int,
        n_layer: int,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.time_mix = RWKVTimeMix(d_model, layer_id, n_layer)
        self.channel_mix = RWKVChannelMix(d_model, d_ff, layer_id, n_layer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RWKV block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.dropout(self.time_mix(self.ln1(x)))
        x = x + self.dropout(self.channel_mix(self.ln2(x)))
        return x


class RWKVModel(nn.Module):
    """
    Complete RWKV language model, interface-compatible with AGI2Model.

    Args:
        config: AGI2Config object. n_embd, n_layer, n_inner, vocab_size and
            the dropout/layer-norm settings are used; n_head and n_positions
            are ignored (RWKV has no attention heads or position embeddings).
    """

    def __init__(self, config: AGI2Config):
        super().__init__()
        self.config = config

        self.token_embeddings = TokenEmbeddings(
            config.vocab_size, config.n_embd, config.embd_pdrop
        )
        # Extra layer norm on the embeddings, as in the reference RWKV.
        self.ln_in = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.blocks = nn.ModuleList(
            [
                RWKVBlock(
                    config.n_embd,
                    config.n_inner,
                    layer_id,
                    config.n_layer,
                    config.resid_pdrop,
                    config.layer_norm_epsilon,
                )
                for layer_id in range(config.n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if config.tie_word_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                config.n_embd, config.vocab_size, bias=False
            )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights; time_* parameters keep
        their schedule-based initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _run_transformer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run input through embeddings, RWKV blocks, and final layer norm.

        Named for interface compatibility with AGI2Model (generation.py calls
        this to obtain hidden states).

        Returns hidden states of shape (batch_size, seq_len, n_embd).
        """
        x = self.ln_in(self.token_embeddings(input_ids))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def _project_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        if self.output_projection is not None:
            return self.output_projection(hidden_states)
        return torch.matmul(hidden_states, self.token_embeddings.embedding.weight.t())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits. Used for generation.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        return self._project_to_logits(self._run_transformer(input_ids))

    def forward_hidden(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both logits and hidden states. Used for training.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)

        Returns:
            Tuple of (logits, hidden_states) where:
            - logits: (batch_size, seq_len, vocab_size)
            - hidden_states: (batch_size, seq_len, n_embd)
        """
        hidden_states = self._run_transformer(input_ids)
        return self._project_to_logits(hidden_states), hidden_states

    def get_num_params(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
