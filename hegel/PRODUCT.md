# Hegel: a transformer compiled from a Markov chain

## What this is

An experiment in building a little language model (LtLM) with **zero training**.
Every weight is computed directly from statistics of the corpus. No gradient
step is ever taken.

The source of those statistics is an adaptive Markov chain over BPE tokens. The
chain is compressed, then written into the standard parts of a transformer:
token embedding, attention heads, feed-forward layers, output decoding. The
model's size is derived from the data, not chosen up front.

## The two questions

1. **Can we do it?** Does a transformer whose weights were written from a
   Markov table behave pretty closely like that Markov chain?
2. **What doors does that open?** Once a working language model can be produced
   by construction, what becomes possible that training doesn't offer?

This is not a comparison against a trained model. Whether the compiled model
beats or loses to SGD is immaterial here.

## What "works" means

- The compiled transformer's next-token distribution is **pretty close** to the
  chain's across the corpus. Exact fidelity is not a goal; the compression step
  is expected to cost something.
- Text sampled from the transformer reads like text sampled from the chain.
- It runs as an ordinary transformer forward pass: no lookup tables on the
  side, no special-case code at inference.

Closeness is reported as numbers (divergence from the chain, top-1 agreement)
plus side-by-side samples. No pass threshold is fixed yet; picking one is part
of the first run.

## Pipeline

| Step | What it does | Output |
|---|---|---|
| 1. Tokenize | Learn a BPE vocabulary on the corpus, small enough that the table stays tractable | token stream, vocab size V |
| 2. Chain | Build an adaptive Markov chain over the whole token stream, in the spirit of `../yuwakisaweb/markov` | context → next-token counts |
| 3. Compress | Shrink the table to what the data supports | a pruned, factored table and the model dimensions |
| 4. Compile | Write the compressed table into transformer weights | a model checkpoint |
| 5. Verify | Compare transformer to chain; sample from both | fidelity numbers, sample text |

### How the table becomes a transformer

- **Embedding and output decoding** come from factoring the next-token
  statistics. The rank kept sets `d_model`.
- **Attention heads** fetch context: head *k* attends to the token *k*
  positions back and copies it into the residual stream.
- **Feed-forward neurons** are the table rows. Each one recognises a context
  and writes that context's contribution to the next-token prediction. The
  number of contexts kept sets `d_ff`.
- Longer contexts refine shorter ones, so the layers stack as corrections.

### Decisions so far

- **One stream.** The corpus is read as a single unbroken token stream, as
  with the "remove newlines" option in yuwakisaweb. Newlines become spaces.
  There are no paragraph groups, no Start/End markers and no resets, so every
  position is predicted the same way.
- **Longest-match backoff.** The prediction comes from the longest context
  that matches, with shorter contexts as fallback. This replaces the original
  chain's summing of counts across context lengths, which a softmax output
  cannot express directly. Summing may be revisited later.
- **Single-token suffixes.** The original chain can emit several tokens per
  step; a transformer emits one, so the chain stores one next token.

### What "compress" means here

Bitwise coding (Huffman, gz) yields an opaque stream that a transformer cannot
compute with, so it is used only as a **ruler**: the coded size of the table is
the information budget, and it tells us how big the model should be.

The compression that carries into weights is structural:

- **Pruning:** keep a longer context only if it predicts differently enough
  from its shorter parent. The yuwakisaweb chain's MDL credit assignment
  already does a version of this.
- **Factoring:** approximate the table at low rank. This is lossy, and it is
  also the only place the model can generalise beyond the chain.

## Scope

**In**

- One corpus, one compiled model, one verification report.
- A minimal transformer of our own in `hegel/`, shaped to be easy to construct
  (the `semantic/` model's LayerNorm, GELU and learned positions fight
  hand-written weights).
- Reuse of `semantic/src/bpe_tokenizer.py` for step 1.
- A Python port of the chain, so chain and compiler share one definition.

**Out**

- Training, fine-tuning, or any gradient step.
- Benchmarks against trained models.
- Exact reproduction of the chain.
- Loading the compiled weights into the `semantic/` model.

## First milestone

The smallest thing that answers question 1:

- Corpus: `samples/corpus20.txt` (about 127 KB).
- Small BPE vocabulary, context length up to 3 tokens.
- Compile, then print fidelity numbers and a page of samples from the chain
  and from the transformer.

## Doors to look behind (question 2)

These are hypotheses to test once question 1 is answered, not claims.

- **Models without a GPU.** A language model for a small corpus in seconds on
  a CPU.
- **Size from data.** The corpus dictates `d_model`, `d_ff`, head count and
  depth, replacing guesswork.
- **Editable models.** Adding text adds neurons; removing text removes them.
  No retraining, and forgetting is exact.
- **Provenance.** Every neuron traces back to specific corpus n-grams.
- **Known circuits.** A model whose mechanism is known by construction is a
  ground truth for interpretability tools.
- **Generalisation from squeezing.** Does a harder low-rank squeeze make the
  model predict sensibly in contexts the chain never saw?
- **A starting point.** Compiled weights as initialisation for training, or as
  a fixed base under trained layers.

## Open questions

- **MDL credit under backoff.** The original chain's credit assignment was
  designed for summed counts, where it preserves what each context length
  uniquely contributes. Whether it carries over unchanged to backoff, or
  pruning needs its own rule, is to be worked out in the chain port.
- **Unseen continuations.** Zero-count next tokens need a floor so the output
  stays finite.
- **How close is "pretty close"?** To be set after seeing the first numbers.
- **Vocabulary size and context length.** Both drive table size; the right
  trade for a 127 KB corpus is unknown.
