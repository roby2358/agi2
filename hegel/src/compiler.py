"""
Compile a chain into transformer weights. No gradient step is taken.

This is the wide (one-hot) build, exact up to float rounding.

Residual stream layout, V = vocab size, n = max context:
    slot 0          [0, V)            the current token, one-hot
    slot k          [k*V, (k+1)*V)    the token k positions back, k = 1..n-1
    logit region    [n*V, (n+1)*V)    next-token logits
"""

import torch
from src.chain import Chain
from src.model import HegelModel

# Score gap between a head's target offset and everything else. exp(-30) is
# below float32 resolution, so attention is hard in practice.
ATTENTION_GAIN = 30.0


def compile_wide(chain: Chain) -> HegelModel:
    """
    Write a chain into a one-hot transformer.

    - Head k attends k positions back and copies that token into slot k. Near
      the start of the stream it attends to the sink and leaves slot k empty.
    - One feed-forward neuron per kept context is an AND gate over the slots
      the context covers: it outputs 1 when every token matches, else 0.
    - Each neuron writes its context's log-probabilities minus those of the
      context it backs off to. When a long context matches, the kept suffixes
      it backs off through all fire too, and the sum telescopes to the
      longest match. The unigram log-probabilities are the output bias.
    """
    vocab = chain.vocab_size
    slots = chain.max_context
    n_heads = slots - 1
    logit = slots * vocab
    model = HegelModel(
        vocab_size=vocab,
        d_model=(slots + 1) * vocab,
        n_heads=n_heads,
        d_head=vocab,
        d_ff=len(chain.contexts) - 1,
        max_offset=n_heads,
    )
    identity = torch.eye(vocab)
    log_probs = torch.from_numpy(chain.log_probs)

    model.embed[:, :vocab] = identity
    model.unembed[:, logit:] = identity
    model.b_out[logit:] = log_probs[0].float()

    for head in range(n_heads):
        back = head + 1
        model.rel_bias[head] = -ATTENTION_GAIN
        model.rel_bias[head, back] = ATTENTION_GAIN
        model.w_v[head, :, :vocab] = identity
        model.w_o[head, back * vocab : (back + 1) * vocab, :] = identity

    for neuron, context in enumerate(chain.contexts[1:]):
        row = neuron + 1
        for slot, token in enumerate(reversed(context)):
            model.w_in[neuron, slot * vocab + token] = 1.0
        model.b_in[neuron] = 1.0 - len(context)
        delta = log_probs[row] - log_probs[chain.parents[row]]
        model.w_out[logit:, neuron] = delta.float()

    return model
