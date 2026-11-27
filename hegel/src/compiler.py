"""
Compile a chain into transformer weights. No gradient step is taken.

Residual stream layout, n = max context, d = slot width, r = logit width:
    slot 0          [0, d)            the current token's vector
    slot k          [k*d, (k+1)*d)    the token k positions back, k = 1..n-1
    logit region    [n*d, n*d + r)    next-token logits, or their code

The wide build has d = r = vocab size, with one-hot token vectors and an
identity decoder; it is exact up to float rounding. The squeezed build uses
short random token vectors in the slots, a narrow logit code, or both.
"""

import numpy as np
import torch
from src.chain import Chain
from src.factor import factor_table, random_codes
from src.model import HegelModel

# Score gap between a head's target offset and everything else. exp(-30) is
# below float32 resolution, so attention is hard in practice.
ATTENTION_GAIN = 30.0


def compile_chain(
    chain: Chain,
    slot_dim: int = 0,
    logit_dim: int = 0,
    match_threshold: float = 0.7,
    low_weight: float = 0.003,
    decoder_rounds: int = 4,
    code_seed: int = 0,
) -> HegelModel:
    """
    Write a chain into a one-layer transformer.

    - Head k attends k positions back and copies that token's vector into
      slot k. Near the start of the stream it attends to the sink and leaves
      slot k empty.
    - One feed-forward neuron per kept context scores the slots it covers
      against its tokens' vectors. A full match scores the context length.
      The neuron outputs 1 on a full match and 0 once the score is short by
      1 - match_threshold or more. With one-hot vectors that is an exact AND
      gate; with short random vectors a near miss scores a little above
      length - 1, and the threshold is what keeps it from firing.
    - Each neuron writes its context's output minus that of the context it
      backs off to. When a long context matches, the kept suffixes it backs
      off through fire too, and the sum telescopes to the longest match. The
      unigram output is the output bias.

    Args:
        chain: The chain to compile.
        slot_dim: Width of each token vector; 0 means one-hot (vocab size).
        logit_dim: Width of the logit code; 0 means the full vocabulary with
            exact log-probabilities.
        match_threshold: How far above length - 1 a score must be before the
            neuron fires at all, between 0 and 1. Irrelevant when one-hot.
        low_weight: What a token with no probability still counts for in the
            output factoring. Used only when logit_dim > 0.
        decoder_rounds: Alternating least-squares rounds for the output
            factoring; 0 keeps a random decoder.
        code_seed: Seed for the random token vectors and decoder.
    """
    vocab = chain.vocab_size
    slots = chain.max_context
    n_heads = slots - 1

    if slot_dim > 0:
        token_codes = torch.from_numpy(random_codes(vocab, slot_dim, code_seed)).float()
        threshold = match_threshold
    else:
        token_codes = torch.eye(vocab)
        threshold = 0.0
    width = token_codes.shape[1]

    if logit_dim > 0:
        # Contexts count by how often they predict, plus 1 so that rare ones
        # (and the unigram row, which only predicts off-corpus) still count.
        matched = chain.match_counts().astype(np.float64)
        row_weights = matched / matched.mean() + 1.0
        decoder, codes = factor_table(
            chain.log_probs,
            logit_dim,
            low_weight,
            row_weights,
            decoder_rounds,
            code_seed + 1,
        )
        outputs = torch.from_numpy(codes)
        unembed = torch.from_numpy(decoder).float()
    else:
        outputs = torch.from_numpy(chain.log_probs)
        unembed = torch.eye(vocab)
    rank = unembed.shape[1]

    logit = slots * width
    model = HegelModel(
        vocab_size=vocab,
        d_model=logit + rank,
        n_heads=n_heads,
        d_head=width,
        d_ff=len(chain.contexts) - 1,
        max_offset=n_heads,
    )
    model.embed[:, :width] = token_codes
    model.unembed[:, logit:] = unembed
    model.b_out[logit:] = outputs[0].float()

    identity = torch.eye(width)
    for head in range(n_heads):
        back = head + 1
        model.rel_bias[head] = -ATTENTION_GAIN
        model.rel_bias[head, back] = ATTENTION_GAIN
        model.w_v[head, :, :width] = identity
        model.w_o[head, back * width : (back + 1) * width, :] = identity

    gain = 1.0 / (1.0 - threshold)
    parents = torch.tensor(chain.parents[1:], dtype=torch.long)
    model.w_out[logit:, :] = (outputs[1:] - outputs[parents]).float().T
    for neuron, context in enumerate(chain.contexts[1:]):
        for slot, token in enumerate(reversed(context)):
            model.w_in[neuron, slot * width : (slot + 1) * width] = (
                gain * token_codes[token]
            )
        model.b_in[neuron] = -gain * (len(context) - 1 + threshold)

    return model


def compile_wide(chain: Chain) -> HegelModel:
    """The exact one-hot build: compile_chain with no squeezing."""
    return compile_chain(chain)
