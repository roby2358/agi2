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
| 2. Chain | Build an adaptive Markov chain over the whole token stream, in the spirit of `../../yuwakisaweb/markov` | context → next-token counts |
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
  with the "remove newlines" option in yuwakisaweb. Newlines become spaces and
  the corpus's `<|endoftext|>` markers are removed. There are no paragraph
  groups, no Start/End markers and no resets, so every position is predicted
  the same way.
- **Longest-match backoff.** The prediction comes from the longest context
  that matches, with shorter contexts as fallback. This replaces the original
  chain's summing of counts across context lengths, which a softmax output
  cannot express directly. Summing may be revisited later.
- **Single-token suffixes.** The original chain can emit several tokens per
  step; a transformer emits one, so the chain stores one next token.
- **Raw counts.** The original chain's MDL credit assignment exists to stop
  summed counts double-counting. Under backoff only one context predicts at a
  time, so it is dropped and pruning has its own rule (below).
- **Floor for unseen tokens.** Every context's distribution is mixed with a
  uniform one at weight 1e-4, so every log-probability is finite.
- **The pruned chain is the reference.** The transformer is compared with the
  chain it was compiled from. What pruning cost is reported separately,
  against the unpruned chain.
- **Wide build first.** A one-hot build that should match the chain exactly
  comes before any lossy squeeze, so that later infidelity is known to come
  from the squeeze and not from a bug.

### What "compress" means here

Bitwise coding (Huffman, gz) yields an opaque stream that a transformer cannot
compute with, so it is used only as a **ruler**: the coded size of the table is
the information budget, and it tells us how big the model should be.

The compression that carries into weights is structural:

- **Pruning:** keep a longer context only if it predicts differently enough
  from the shorter context it would back off to. The measure is the
  log-likelihood the corpus would lose without it: occurrences times KL
  divergence, against a threshold in nats.
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

### Result

Done, with the wide one-hot build. On corpus20 (54,944 tokens, vocabulary 512,
context 3), pruning at 0.5 nats keeps 24,419 of 34,075 contexts for a mean
cost of 0.010 nats per position against the unpruned chain. The compiled
transformer (`d_model` 2048, 2 heads, `d_ff` 24,419) then matches that chain
at every position: mean KL 3e-7 nats, 100% top-1 agreement, and samples that
are identical token for token. The whole run takes about 40 s on a CPU.

So question 1 is answered yes for the exact build. It is not yet answered for
a compressed one.

## Second milestone

The squeezed build: replace one-hot slots with short token vectors and the
V-wide logit region with a lower-rank one, then report the same numbers.

First scratch measurements say the two halves behave very differently:

- The input side squeezes almost for free. Random 32-dimensional token
  vectors with a raised firing threshold reproduce the chain to within 1e-4
  nats.
- The output side does not squeeze by a plain SVD of the log-probability
  table. Rank 256 of a possible 330 or so still costs about 0.3 nats, and
  rank 64 costs 3.4. A random decoder does better at low rank (0.76 nats at
  rank 64) but stalls at about 87% top-1 agreement.

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

- **How to squeeze the output side.** The chain's distributions are nearly
  one-hot, the worst case for a least-squares low-rank fit in log space. What
  closed-form factoring respects the softmax, without a gradient step?
- **How close is "pretty close"?** The exact build needs no threshold. One is
  needed for the squeezed build; the pruning sweep shows cost rising steadily
  with no knee, so it will be a choice and not a discovery.
- **Vocabulary size and context length.** Both drive table size. 512 and 3
  work; nothing else has been tried.
- **Summing versus backoff.** Whether the original chain's summing across
  context lengths is worth expressing.
- **In-sample only.** All numbers so far are on the corpus the chain was
  built from, where it mostly recalls (0.53 nats per token). Nothing yet says
  how chain or transformer behave on held-out text.

Settled: credit assignment under backoff (dropped) and the floor for unseen
tokens (uniform mixing at 1e-4); both are under "Decisions so far".
