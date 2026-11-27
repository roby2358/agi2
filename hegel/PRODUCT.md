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
- **Held-out split.** A random 10% of the text, sampled as blocks of about
  1000 characters, is kept from the tokenizer and the chain and used only for
  scoring.
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

So question 1 is answered yes for the exact build. The second milestone
answers it for a compressed one.

## Second milestone

The squeezed build: replace one-hot slots with short token vectors and the
V-wide logit region with a lower-rank code, then report the same numbers.

### Result

Done. With 32-dimensional random token vectors and a 128-dimensional logit
code, `d_model` falls from 2048 to 224 and the weight count from 106 M to
11 M. Against the same chain, over all corpus positions: mean KL 0.0022
nats, 99.83% top-1 agreement, and the model's own score on the corpus moves
from 0.5364 to 0.5389 nats per token. Samples read like the chain's.

The two halves behaved very differently:

- **The input side squeezes almost for free.** Random token vectors work as
  long as a context neuron only fires when its match is nearly complete.
  With a firing threshold, 24 to 32 dimensions are indistinguishable from
  one-hot. Without one, thousands of near-miss contexts fire and the output
  is noise.
- **The output side needed a different kind of factoring.** A plain SVD of
  the log-probability table is hopeless: rank 128 costs 1.9 nats, because the
  chain's distributions are nearly one-hot and an even-handed fit spends its
  rank on the entries that say "never". What works is a weighted fit that
  gets each context's seen tokens right and only keeps the unseen ones low,
  solved by alternating closed-form least squares between context codes and
  token decoder vectors.

Rank is a smooth dial: a 64-dimensional code costs 0.02 nats and 98.3%
top-1, 32 costs 0.2 nats and 88.8%.

One judgement call to note: alternating least squares iterates, though each
step is a closed-form solve on the table and the model is never run. The
strictly one-shot version (random decoder, one solve per context) costs
about 0.018 nats at rank 128, and it mishandles contexts with many possible
continuations.

## Using it

`hegel_generate.py` (or `just hegel-run "source string"`) compiles a model
from a config and continues a source string; with no string it prompts
interactively.

## Held-out evaluation

A random 10% of the corpus, sampled as blocks of about 1000 characters, is
kept away from the tokenizer and the chain and used only for scoring. Two
findings.

**The compile holds up on unseen text.** The wide build still matches its
chain exactly. The squeezed build stays close: mean KL 0.016 nats and 97.9%
top-1 agreement, against 0.002 and 99.8% on the training text.

**The chain does not.** It scores 0.51 nats per token on its training text
and 5.88 on held-out text. That is worse than unigram frequencies (5.32) and
far worse than a chain with a single token of context (3.99); uniform
guessing is 6.24. The cause is hard backoff with no hedging: a context seen
once predicts its one continuation with near certainty, and about a third of
held-out tokens land on the floor. Longer contexts make this worse, the
reverse of what they do in-sample.

Within the current design a much larger floor (0.1) and much harder pruning
bring it to about 3.17. Mixing each context's distribution with the shorter
context it backs off to (standard interpolated smoothing) reaches 2.66 in a
scratch test, and makes longer contexts useful again.

A curiosity: the squeezed transformer scores slightly better than its own
chain on held-out text, 5.57 against 5.88. Its approximation error leaks a
little probability onto tokens the chain forbids, which acts as a softer
floor. It is an accident, not generalisation.

A caution: on a corpus this small one split is a noisy measure. Six random
splits put the chain between 4.65 and 5.88, and the shipped seed is the
hardest of them. The ordering of methods is the same in every split.

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

- **Smoothing the chain.** Held-out evaluation shows the chain, not the
  compile, is the weak link. Interpolating each context with its parent is
  a change to the chain alone and the compiler does not care how a row was
  made. But smoothed rows are dense, and the output factoring currently
  relies on rows having few seen tokens. Does the squeeze survive it?
- **Shrinking `d_ff`.** After the squeeze, the feed-forward layer (one neuron
  per kept context, 24,419 of them) is nearly all of the model. Pruning is
  the only lever on it so far, and it costs fidelity steadily.
- **Structured token vectors.** Random vectors make near-miss firing pure
  noise, to be suppressed. Vectors that put similar tokens close together
  could make it useful: a context neuron would fire for a context it never
  saw. That is the route to generalisation, and it is untried.
- **How close is "pretty close"?** Both the pruning sweep and the rank sweep
  show cost rising steadily with no knee, so it is a choice and not a
  discovery. The current defaults cost 0.010 nats (pruning) plus 0.002
  (squeeze).
- **Vocabulary size and context length.** Both drive table size. 512 and 3
  work; nothing else has been tried.
- **Summing versus backoff.** Whether the original chain's summing across
  context lengths is worth expressing.

Settled: credit assignment under backoff (dropped), the floor for unseen
tokens (uniform mixing at 1e-4), and how to squeeze the output side (weighted
alternating least squares).
