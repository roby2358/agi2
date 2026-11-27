"""
A minimal one-layer transformer shaped for hand-written weights.

Token embedding, positional attention with a zero sink, a ReLU feed-forward
layer, and a linear decoder with softmax. There is no LayerNorm, no learned
absolute position and no dropout. Every parameter starts at zero and is
filled in by the compiler; nothing here is ever trained.
"""

import torch
import torch.nn as nn


class HegelModel(nn.Module):
    """
    One attention layer followed by one feed-forward layer.

    Attention scores depend only on relative position: head h scores a key
    `offset` positions back with rel_bias[h, offset], and cannot see keys
    further back than max_offset or ahead of the query. There are no query
    or key matrices (equivalent to setting them to zero). Every query also
    sees one extra all-zero key with score 0, the sink (PyTorch's
    add_zero_attn): a head whose target does not exist yet attends there and
    so writes nothing.

    Args:
        vocab_size: Number of token ids.
        d_model: Residual stream width.
        n_heads: Number of attention heads.
        d_head: Width of each head's value vector.
        d_ff: Number of feed-forward neurons.
        max_offset: Furthest position back that attention can score.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        max_offset: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.max_offset = max_offset

        def weight(*shape: int) -> nn.Parameter:
            return nn.Parameter(torch.zeros(*shape), requires_grad=False)

        self.embed = weight(vocab_size, d_model)
        self.rel_bias = weight(n_heads, max_offset + 1)
        self.w_v = weight(n_heads, d_head, d_model)
        self.w_o = weight(n_heads, d_model, d_head)
        self.w_in = weight(d_ff, d_model)
        self.b_in = weight(d_ff)
        self.w_out = weight(d_model, d_ff)
        self.b_out = weight(d_model)
        self.unembed = weight(vocab_size, d_model)

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        """What the attention heads add to the residual stream."""
        batch, length, _ = x.shape
        position = torch.arange(length, device=x.device)
        offset = position[:, None] - position[None, :]
        visible = (offset >= 0) & (offset <= self.max_offset)
        scores = self.rel_bias[:, offset.clamp(0, self.max_offset)]
        scores = scores.masked_fill(~visible, float("-inf"))
        sink_score = scores.new_zeros(self.n_heads, length, 1)
        weights = torch.softmax(torch.cat([sink_score, scores], dim=-1), dim=-1)

        sink = x.new_zeros(batch, 1, self.d_model)
        keys = torch.cat([sink, x], dim=1)
        values = torch.einsum("hed,bjd->bhje", self.w_v, keys)
        mixed = torch.einsum("htj,bhje->bhte", weights, values)
        return torch.einsum("hde,bhte->btd", self.w_o, mixed)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Next-token log-probabilities at every position.

        Args:
            tokens: Token ids, shape (batch, length).

        Returns:
            Log-probabilities, shape (batch, length, vocab_size).
        """
        x = self.embed[tokens]
        x = x + self.attend(x)
        hidden = torch.relu(x @ self.w_in.T + self.b_in)
        x = x + hidden @ self.w_out.T + self.b_out
        return torch.log_softmax(x @ self.unembed.T, dim=-1)

    def parameter_count(self) -> int:
        """Total number of weights, zeros included."""
        return sum(p.numel() for p in self.parameters())

    def nonzero_count(self) -> int:
        """Number of weights the compiler actually set."""
        return sum(int(torch.count_nonzero(p)) for p in self.parameters())
