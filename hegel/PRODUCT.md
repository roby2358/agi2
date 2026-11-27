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
- **Longest-match backoff, smoothed.** The prediction comes from the longest
  context that matches, with shorter contexts as fallback. What that context
  predicts is its own frequencies mixed with the shorter contexts'
  distributions. This stands in for the original chain's summing of counts
  across context lengths, which a softmax output cannot express directly.
- **Single-token suffixes.** The original chain can emit several tokens per
  step; a transformer emits one, so the chain stores one next token.
- **Raw counts.** The original chain's MDL credit assignment exists to stop
  summed counts double-counting. Under backoff only one context predicts at a
  time, so it is dropped and pruning has its own rule (below).
- **Floor for unseen tokens.** The unigram distribution is mixed with a
  uniform one at weight 1e-4, and smoothing passes that down to every
  context, so every log-probability is finite.
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

(Measured with the original hard-backoff chain built on the whole corpus;
the same holds for the second milestone's numbers. Current numbers are under
the third milestone.)

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

`hegel_generate.py` (or `just run "source string"` from `hegel/`) compiles a model
from a config and continues a source string; with no string it prompts
interactively. For a large corpus, `just build hegel` compiles once and saves
the weights and `just chat hegel` talks to them without recompiling.

## Held-out evaluation

A random 10% of the corpus, sampled as blocks of about 1000 characters, is
kept away from the tokenizer and the chain and used only for scoring. The
chain is scored on five different splits, because one split of a corpus this
small swings by more than a nat. The model used for generation is built on
the whole corpus; the split is for choosing settings.

**The compile holds up on unseen text.** The wide build matches its chain
exactly there too. The squeezed build is as close on held-out text as on
training text.

**The original chain did not.** With hard backoff it scored 0.51 nats per
token on its training text and 5.29 held out, about as bad as unigram
frequencies (5.32) and far worse than a chain with one token of context
(3.45). Its best guess was right as often as any other version's, 49% of the
time. The trouble was certainty: a context seen once predicted its one
continuation with near certainty, and a third of held-out tokens landed on
the floor.

## Third milestone: a chain that hedges

Each context's distribution is now mixed with the distribution of the
context one token shorter (Witten-Bell smoothing, which has no tunable: a
context's own share grows with how often it was seen and shrinks with how
many different continuations it had). The longest match still predicts
alone; only what it says changed. Nothing in the compiler changed, since a
neuron writes whatever distribution its context holds.

### Result

Held-out, mean of five splits:

| predictor | nats per token | top-1 accuracy |
|---|---|---|
| unigram | 5.32 | |
| chain, 1 token of context | 3.45 | 17% |
| hard backoff, 3 tokens | 5.29 | 49% |
| smoothed, 3 tokens, as shipped | 2.47 | 48% |

- Longer contexts help again, up to 3 tokens. A fourth adds nothing on this
  corpus.
- Pruning now barely affects held-out quality: keeping a quarter of the
  contexts costs 0.06 nats. It costs fluency on the training text much more,
  so the shipped threshold stays modest (17,812 contexts kept of 31,873).
- The squeeze survives, at a price. Smoothed distributions are dense, so the
  output factoring was reworked to weight each token by its probability.
  At `d_model` 224 the squeezed build is within 0.012 nats of its chain and
  agrees on the top token 99.0% of the time (0.002 and 99.8% under hard
  backoff).
- Samples at full temperature read rougher than before, because the chain
  now gives real probability to off-script tokens. Generation uses a
  temperature of 0.8.

The shipped squeezed model: 8.3 M weights, 2.62 nats per token on held-out
text.

## Fourth milestone: smoothing that knows what it has not seen

Witten-Bell trusted rare contexts too much: a context seen once got half
the say. Top-token accuracy kept rising with longer contexts while the
score got worse, on every corpus. Kneser-Ney smoothing replaces it. Every
count gives up a fixed discount to the shorter context, and shorter
contexts count how many different things preceded them, not how often they
occurred. Again nothing in the compiler changed.

### Result

Held-out nats per token, 3 tokens of context, as shipped:

| corpus | Witten-Bell | Kneser-Ney | contexts kept, before and after |
|---|---|---|---|
| corpus20 (mean of 5 splits) | 2.47 | 2.37 | 17,915 to 11,370 |
| Moby Dick (mean of 3 splits) | 3.06 | 2.90 | 135,713 to 47,011 |
| Hegel (one split) | 2.43 | 2.35 | 372,721 to 195,534 |

- The models are a third to a half smaller, because the same pruning rule
  finds fewer contexts worth keeping, and they predict better.
- The squeezed transformer follows the new chain at least as closely: it
  agrees on the top token 99.3% to 99.5% of the time on all three corpora.
- The saved Hegel model is now 94 M weights (375 MB), built in 5 minutes.
- Longer context now pays. On Hegel the unpruned chain scores 2.34 with 3
  tokens, 2.29 with 4 and 2.26 with 6. That is measured on the chain only;
  no transformer with more than 3 tokens of context has been compiled.
- On the small corpus pruning now costs more (0.06 nats) and changes the
  top token on 14% of the training text. The threshold has not been
  re-tuned.

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

- **Shrinking `d_ff`.** After the squeeze, the feed-forward layer (one neuron
  per kept context, 11,393 of them on corpus20) is nearly all of the model. Pruning is
  the only lever on it so far. It is cheap on held-out text and costly on
  the training text.
- **Does token similarity pay anywhere?** Vectors that put similar tokens
  close together were tried, so that a context neuron would fire for a
  context it never saw. On this corpus it did not help at any setting: the
  best case equals the chain, and looser thresholds are much worse, because
  several similar contexts fire together and their corrections add up.
  Blending similar contexts' distributions inside the chain instead shows a
  small real gain (about 0.03 nats beyond plain extra hedging), too small
  to build. A larger corpus did not change the answer: on Moby Dick (ten
  times the text) with a 2048-token vocabulary the vectors are clearly
  meaningful (" sea" sits next to " whale", " fish" and " ship"), firing
  for similar contexts still only hurts, and blending inside the chain is
  worth about 0.1 nats of 4.6.
- **Can anything reach past the context?** A head that averages the last 50
  tokens, with a table counted from the corpus saying which tokens raise
  which, was measured before building. It carries real signal (a window
  from the wrong place gains nothing) but only 0.01 to 0.03 nats, on
  corpus20 and on Moby Dick. Not built. Samples still have no thread
  longer than three tokens, and this is not the way to get one.
- **The chain is overconfident.** Dividing its logits by 1.2 gains 0.07 to
  0.14 nats held out, more than anything else tried, and needs no new
  parts. Measured on one split; not yet built.
- **Does it scale?** Moby Dick (ten times the text) compiles in about 3
  minutes on a CPU: 47,062 contexts, 21 M weights, and the squeezed model
  agrees with its chain on the top token 99.3% of the time. Hegel (a
  hundred times) builds in 5 minutes: 207,974 contexts, 94 M weights.
- **How close is "pretty close"?** Both the pruning sweep and the rank sweep
  show cost rising steadily with no knee, so it is a choice and not a
  discovery. On held-out text the current defaults cost 0.013 nats
  (pruning) and nothing measurable (squeeze).
- **Vocabulary size and context length.** Both drive table size. 512 and 3
  work. Context 4 to 6 improves the chain under Kneser-Ney but has not been
  compiled; one idea is a single block applied in a loop, one pass per
  token of context.

Settled: credit assignment under backoff (dropped), the floor for unseen
tokens (uniform mixing at 1e-4), how to squeeze the output side (weighted
alternating least squares), and summing versus backoff (smoothing mixes
context lengths inside each context's own distribution, which a softmax can
express).
