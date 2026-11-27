# CLAUDE.md

Guidance for Claude Code when working in `hegel/`. The repo-root `CLAUDE.md`
covers the shared environment and the sibling `semantic/` experiment.

## Keep this file current

This file is the working memory of the experiment. Update it in the same change
whenever any of these happen:

- A design decision is made or reversed.
- A construction note below turns out wrong, or is confirmed by running code.
- Files, commands or dimensions are added or change.
- An open question is settled (also move it in `PRODUCT.md`).

Replace stale text; do not append a changelog. `PRODUCT.md` says what and why,
this file says how.

## What this is

A transformer compiled from an adaptive Markov chain with **zero training**.
Read `PRODUCT.md` first. The test is whether the compiled model behaves pretty
closely like the chain, not whether it beats a trained model.

## Status

Two builds run end to end on `../samples/corpus20.txt`, on a Witten-Bell
smoothed chain. A random 10% of the text is held out: 49,416 training
tokens, 5,540 held-out tokens, V = 512, context 3, prune threshold 0.5,
17,812 contexts kept of 31,873. Fidelity is the transformer against the
chain it was compiled from.

| | wide (`corpus20`) | squeezed (`corpus20_squeezed`) |
|---|---|---|
| slots / logit region | one-hot 512 / exact 512 | random 32-dim / 128-dim code |
| `d_model`, `d_ff` | 2048, 17,812 | 224, 17,812 |
| weights | 79 M (9.2 M nonzero) | 8.3 M (3.8 M nonzero) |
| training text: mean KL (nats) | below 1e-6 | 0.012 |
| training text: top-1 agreement | 100% | 99.0% |
| training text: NLL (chain 0.884) | 0.884 | 0.897 |
| held-out: mean KL (nats) | below 1e-6 | 0.012 |
| held-out: top-1 agreement | 100% | 98.5% |
| held-out: NLL (chain 2.662) | 2.662 | 2.620 |

The chain as a language model, held-out NLL in nats per token, mean over 5
random splits with the range in brackets:

| predictor | NLL | top-1 accuracy |
|---|---|---|
| uniform | 6.24 | |
| unigram | 5.32 | |
| chain, 1 token of context | 3.45 [3.37, 3.51] | 17.3% |
| hard backoff, context 3 (`smoothing = "none"`) | 5.29 [4.65, 5.88] | 48.6% |
| smoothed, context 3, unpruned | 2.45 [2.21, 2.66] | 48.8% |
| smoothed, context 3, as configured | 2.47 [2.23, 2.66] | 47.7% |

`hegel_generate.py` continues a source string with either build, compiled
on the whole corpus.

The squeezed build also compiles on `../samples/moby_dick_pg2701.txt` (ten
times the text) and on `resources/hegel.txt` (a hundred times); see "Moby
Dick compile" and "Hegel build" below. The Hegel weights are saved and can
be used without recompiling.

Two extensions were measured as scratch scripts and not adopted: structured
token vectors and a wide-window head (sections below). Neither moves
held-out NLL by more than a few hundredths of a nat. Dividing the logits by
a temperature of 1.2 gains more than either (see "Wide-window head") and is
the cheapest candidate next step; nothing is chosen yet.

## Files and commands

```
hegel_compile.py        entry point: tokenize, chain, compile, verify, sample
hegel_build.py          compile on the whole corpus and save the weights
hegel_generate.py       source string in, continuation out; or interactive
resources/corpus20.toml           the exact wide build
resources/corpus20_squeezed.toml  32-dim slots, 128-dim logit code
resources/moby_dick.toml          the squeezed build on Moby Dick (about 8 min)
resources/hegel.toml              the squeezed build on resources/hegel.txt
resources/hegel.txt               13 MB of Hegel in English; not in git
compiled/                         saved weights; not in git
src/config.py           HegelConfig, load_config (TOML, unknown keys rejected)
src/corpus.py           normalize (one stream), tokenize (BPE from semantic/)
src/chain.py            count_contexts, build_chain (pruning), Chain (backoff)
src/model.py            HegelModel: the minimal transformer, all weights zero
src/factor.py           random codes; weighted low-rank factoring of the table
src/compiler.py         compile_chain (wide or squeezed), compile_wide
src/pipeline.py         compile_from_config: corpus -> tokens -> chain -> model
src/holdout.py          score the chain on several random held-out splits
src/verify.py           FidelityMeter, chain/model comparisons, coupled sampling
src/checkpoint.py       save_weights, load_weights (model + tokenizer in one file)
src/generation.py       respond: continue a source string
tests/                  one test file per src module
justfile                the compile, build, chat and run recipes
```

```bash
cd hegel
just compile                         # compile + verify corpus20 (wide)
just compile corpus20_squeezed       # the squeezed build
just compile moby_dick               # squeezed, Moby Dick, about 8 min
just run "source string"             # continue a string (wide build)
just run "source string" corpus20_squeezed
just run                             # interactive; empty line quits
just build hegel                     # whole corpus, save compiled/hegel.pt, 7 min
just chat hegel                      # interactive, from the saved weights
uv run python hegel_compile.py resources/corpus20_squeezed.toml
uv run python hegel_generate.py resources/corpus20.toml "source string"
uv run pytest tests        # 105 tests, about 15 s
uv run black src tests hegel_compile.py hegel_generate.py hegel_build.py
uv run isort src tests hegel_compile.py hegel_generate.py hegel_build.py
uv run flake8 --max-line-length 88 --extend-ignore E203 src tests *.py
uv run mypy src hegel_compile.py hegel_generate.py hegel_build.py
```

Run from `hegel/`: scripts and tests import `src.*` relative to it.

Three scripts, three jobs. `hegel_compile.py` measures: it compiles on the
training part of a held-out split, checks the transformer against its chain
and saves nothing. `hegel_build.py` compiles on the whole corpus
(`holdout_fraction` forced to 0) and saves the model and its tokenizer to
the config's `checkpoint` path. `hegel_generate.py` loads that file if the
config names one and it exists, and otherwise compiles on the whole corpus
itself (about 15 s wide, 40 s squeezed on corpus20), saving nothing. Only
`hegel.toml` sets `checkpoint`.

The weights file records the settings that decide the weights. If the
config no longer matches, `hegel_generate.py` says so and recompiles
instead of using it. It does not notice a changed corpus file: rebuild after
editing the text.

## Working rules

- **Do not commit or push without the user's explicit OK each time.** They
  review changes first.
- No gradient steps anywhere. If something needs training to work, that is a
  finding to report, not a thing to add.
- Environment is the repo root's `pyproject.toml` / `uv.lock`; run with
  `uv run`. Python 3.11+, black (88), isort (profile black), mypy strict.
- Tokenization reuses `../semantic/src/bpe_tokenizer.py`. Both experiments
  name their package `src`, so `src/corpus.py` loads it by file path. Do not
  reuse `AGI2Model`: its LayerNorm, GELU and learned absolute positions fight
  hand-written weights.
- The reference chain is `../../yuwakisaweb/markov/js/markov.js`.
  `src/chain.py` is the single definition shared by chain and compiler; it
  follows the reference in spirit, with the differences under "Decisions".

## Decisions

- **One stream.** The corpus is one unbroken token stream. `<|endoftext|>`
  markers are removed and every run of whitespace, newlines included, becomes
  one space. No groups, no Start/End markers, no resets.
- **Held-out split.** The normalized text is cut into blocks of about
  `holdout_block` (1000) characters at word boundaries, and a random
  `holdout_fraction` of the blocks (chosen by `holdout_seed`) is held out.
  Neither the tokenizer nor the chain sees them. Training and held-out text
  are each their blocks joined in order, so each has a few seams (about 13
  in the held-out stream) where non-adjacent text meets. Both shipped
  configs hold out 10%. Single tokens cannot be sampled, since a token is
  scored from its context.
- **The held-out split is for choosing settings.** `hegel_compile.py` uses
  it; `hegel_generate.py` builds on the whole corpus with the same settings.
  The chain is scored on `holdout_seeds` (5) different splits and reported
  as mean and range, because one split of this corpus swings by more than a
  nat. The transformer is compiled and checked on the first split only.
- **Longest-match backoff, smoothed.** The longest kept context that matches
  predicts alone. With `smoothing = "witten_bell"` (the shipped setting) what
  it predicts is its own relative frequencies mixed with the distribution of
  the context one token shorter, recursively down to the floored unigram.
  Its own share is n / (n + t) for n occurrences and t distinct
  continuations. The mix always uses the next context down, whether or not
  pruning keeps it. `smoothing = "none"` is hard backoff: each context's own
  frequencies and the floor.
- **Sampling temperature.** `temperature` (0.8 shipped) applies to samples
  and generation only, never to fidelity or NLL numbers. The smoothed chain
  gives real probability to off-script tokens, and at temperature 1 its
  samples read rougher than hard backoff's.
- **Single-token suffixes.** A context maps to one next token.
- **Context length counts the current token.** `max_context = 3` means the
  current token and the two before it, so 2 attention heads.
- **Floor.** `floor = 1e-4` of uniform mass is mixed into the unigram
  distribution (and, with `smoothing = "none"`, into every context's).
- **Raw counts, no credit assignment.** The original chain's MDL credit
  subtraction exists to stop summed counts double-counting. Under backoff only
  one context predicts at a time, so it is dropped.
- **Pruning rule.** Contexts are visited shortest first. A context's gain is
  `occurrences * KL(context || longest kept proper suffix)` in nats, the
  log-likelihood the corpus loses without it. It is kept if the gain reaches
  `prune_threshold`. The kept set is therefore not suffix-closed, and each
  context's parent is its longest kept proper suffix.
- **The pruned chain is the reference.** Model fidelity is measured against
  the chain it was compiled from. The cost of pruning is reported separately,
  against the unpruned chain.
- **Sink position.** Every attention query also sees one all-zero key with
  score 0 (the same thing as PyTorch's `add_zero_attn`). A head whose target
  position does not exist yet attends there and writes nothing. The chain
  never sees it, so it is not a Start marker.
- **Positional attention only.** Scores come from a relative-position bias
  per head. There are no query/key matrices (equivalent to zero ones).
- **Factoring is alternating least squares, not SVD.** The output table is
  factored by alternating closed-form least-squares solves (codes given
  decoder, decoder given codes). This is a judgement call on "no gradient
  steps": it iterates, but it factors the table, never runs the model and
  takes no gradient. `decoder_rounds = 0` gives the strictly one-shot version
  (random decoder, one solve per context), which costs about 8 times the KL.
- **Bitwise coding is a ruler only.** Huffman or gz size of the table gives
  the information budget for sizing the model. The compression that reaches
  the weights is pruning and low-rank factoring. Not implemented yet.

## Construction notes

### Model shape (confirmed)

One layer: positional attention with the sink, then a ReLU feed-forward layer,
then a linear decoder and softmax. No LayerNorm, no absolute positions. The
feed-forward biases do the work a constant-1 residual dimension would, so
there is none.

### Wide build (confirmed, `compile_wide`)

- Residual stream: slot 0 is the current token one-hot (V dims); slot *k* is
  the token *k* back (V dims each, k = 1..n-1); then a logit region (V dims).
  `d_model = V * (n + 1)`.
- **Attention.** Head *k* has bias +30 at offset *k* and -30 elsewhere; its
  value matrix reads slot 0 and its output matrix writes slot *k*. exp(-30) is
  below float32 resolution, so attention is hard in practice.
- **Context neurons.** One per kept context: weight 1 on the context's token
  in each slot it covers, bias `-(length - 1)`. After ReLU it is 1 on a full
  match and 0 otherwise.
- **Telescoping output.** The unigram log-probabilities are the output bias
  on the logit region. The neuron for context *c* writes
  `log P(. | c) - log P(. | parent(c))`. When a context matches, the kept
  suffixes it backs off through all fire, and the sum telescopes to the
  longest match. This holds with gaps in the suffix chain because each delta
  is taken against the longest kept proper suffix (tested with pruning).
- Sizes: `d_ff` = kept contexts, heads = n - 1, `d_head` = V.

### Pruning (measured, smoothed chain, mean of 5 splits)

| threshold (nats) | contexts kept | held-out NLL | held-out top-1 | training NLL |
|---|---|---|---|---|
| 0 | 32,139 | 2.453 | 48.8% | 0.745 |
| 0.2 | 23,922 | 2.457 | 48.8% | 0.789 |
| 0.5 | 17,915 | 2.466 | 47.7% | 0.894 |
| 1 | 12,339 | 2.482 | 46.1% | 1.082 |
| 2 | 7,774 | 2.516 | 43.7% | 1.337 |
| 4 | 4,190 | 2.587 | 40.6% | 1.681 |
| 8 | 2,397 | 2.691 | 37.0% | 1.998 |

- Pruning barely moves held-out NLL: a quarter of the contexts (threshold 2)
  costs 0.06 nats. It costs the training text much more, which is what
  samples are judged by. 0.5 is the shipped compromise.
- Context 4 does not help (2.46 at best); context 2 gives 2.55.
- Under hard backoff the picture was different. In-sample, pruning only
  cost (threshold 0.5: KL 0.010 against the unpruned chain; 2: 0.21; 8:
  0.98, no knee). Held out, pruning helped, because it removed overconfident
  once-seen contexts.

### Squeezed build (confirmed, `compile_chain`)

Same layout as the wide build with narrower regions:
`d_model = n * slot_dim + logit_dim`.

**Input side: random token vectors and a firing threshold.** Each token gets
a random unit vector of `slot_dim` dims. A context neuron takes the dot
product of the slots it covers with its tokens' vectors; a full match scores
the context length, a one-token miss scores length - 1 plus the dot product
of two random vectors. The neuron computes
`relu((score - (length - 1) - tau) / (1 - tau))`, so it is 1 on a full match
and silent unless the miss is closer than `tau` (`match_threshold`).

- This is nearly free: 32 dims at tau 0.7 is indistinguishable from one-hot,
  and 24 dims is the same. 16 dims costs KL 0.017 at tau 0.7 and 0.004 at
  tau 0.8.
- Too low a threshold is catastrophic (32 dims, tau 0.3: KL 4.7), because
  near-miss contexts fire by the thousand.

**Output side: weighted factoring (`src/factor.py`).** The logit region holds
a `logit_dim` code and the decoder has one vector per token. Each context's
target is its log-probabilities minus the row's lowest, so unlikely tokens
sit at 0. Each token's error is weighted by its probability relative to the
row's top token, plus `low_weight` (0.003). Codes and decoder are fitted by
alternating closed-form solves, starting from a random decoder. Contexts are
weighted by how often they predict, plus 1. Neurons write code deltas
against their parent exactly as in the wide build, and telescoping still
holds because it is linear.

Smoothed chain, full model on split 0 (slots 32, tau 0.7, 4 rounds):
rank 128 gives training KL 0.012 and 99.0% top-1, held-out KL 0.012 and
98.5%. From the scratch runs on the factoring alone: rank 64 gives KL 0.076
and 97.6%, rank 192 about 0.005. Factoring takes about 20 s at rank 128.

What was tried on the way, all at rank 128 (mean KL in nats; top-1):

| method | hard backoff | smoothed |
|---|---|---|
| SVD of the log-probability table | 1.9; 72% | |
| random decoder, plain projection | 0.49; 84% | |
| weights 1 on seen tokens, 0.01 on unseen, random decoder | 0.018; 98.8% | |
| the same with alternating least squares | 0.002; 99.8% | 0.019; 97.8% |
| unweighted alternating least squares | | 0.57; 60% |
| probability weights + 0.01 | 0.009; 99.7% | 0.021; 98.2% |
| probability weights + 0.003 (shipped) | 0.0015; 99.96% | 0.012; 99.0% |

- SVD fails because an even-handed fit in log space spends its rank on the
  tokens that do not matter.
- Seen/unseen weighting was built for hard backoff's nearly one-hot rows.
  Smoothed rows are dense (about 48 tokens above the baseline per row), and
  probability weighting handles both.
- With a random decoder and no refit, broad rows leak badly and the unigram
  row is destroyed; under hard backoff that showed up as byte garbage in
  samples. Refitting the decoder fixes it.
- Smoothing costs the squeeze about 8 times the KL at the same rank.
- `decoder_rounds` beyond 4 does not help at rank 128.

### Structured token vectors (measured, not adopted)

Token vectors were built from the training bigram counts: positive PMI of
each token with what follows and what precedes it, centred, reduced by SVD,
unit length. Neighbours look right where a token is a word (" your" is
nearest " their" at 0.91, " or" nearest " to" and " and") and vague where
it is a fragment ("t", "our"), which most of the 512 tokens are. Measured
on the wide build's exact logits, held-out NLL, mean of splits 0 to 2,
chain 2.460 and 47.8% top-1. The experiments were scratch scripts; nothing
in `src/` changed.

| variant | best held-out NLL |
|---|---|
| near-miss neurons fire as compiled, 32 dims, tau 0.7 / 0.9 | 3.25 / 2.484 |
| the same, 64 dims, tau 0.7 / 0.9 | 2.574 / 2.463 |
| near-miss output scaled to 0.2 | 2.464 |
| oracle: only contexts longer than the exact match may fire partially | 2.459 |
| oracle: the same, averaged instead of summed | 2.459 |
| mix 10% of similar contexts' average distribution, in probability space | 2.396 |
| control: the same mix with random vectors (5%) | 2.439 |
| control: mix in 5% unigram | 2.430 |

- **Near-miss firing never helps, at any threshold.** A neuron adds its
  context's correction in log space. Several similar contexts firing at
  once sum their corrections and overshoot, and they also fire on top of an
  exact match that needed no help. Lower thresholds are steadily worse.
- **Removing both faults does not rescue it.** The two oracle rows gate and
  average in ways one ReLU layer cannot, and still gain 0.001 nats.
- **There is a little similarity signal, in probability space.** Mixing in
  the average distribution of similar contexts gains 0.064 nats; the same
  mix with random vectors or plain unigram gains about 0.03, so half of it
  is only extra hedging. Top-1 accuracy does not move in any row. The mix
  weight and threshold were picked on the held-out text, so these gains are
  upper bounds.
- A probability-space mix is not something a neuron can add to logits. It
  would have to go into the chain (each context's distribution smoothed
  toward similar contexts), with the compiler unchanged. For about 0.03
  nats that was not built.
- The unigram control says the smoothed chain is still slightly
  overconfident: 5% more unigram mass is worth 0.03 nats.

The same test on `../samples/moby_dick_pg2701.txt` (ten times the text;
split 0, first 20,000 held-out positions, prune 0.5) gives the same answer.
NLL per token is not comparable across vocabulary sizes.

| | V = 512 | V = 2048 |
|---|---|---|
| training / held-out tokens | 512,011 / 57,072 | 339,510 / 37,871 |
| contexts kept | 135,713 | 193,843 |
| chain | 3.083; 33.6% | 4.607; 22.6% |
| near-miss firing, 64 dims, tau 0.9 | 3.198 | 4.726 |
| near-miss firing, 64 dims, tau 0.7 | 4.49 | 6.30 |
| oracle: gated and averaged | 3.081 | 4.604 |
| probability-space mix, similar contexts (best tried) | 3.011 | 4.421 |
| control: the same mix, random vectors (best tried) | 3.055 | 4.547 |
| control: unigram mix (best tried) | 3.054 | 4.536 |

- At V = 2048 the vectors are plainly meaningful (" sea" is nearest
  " whale", " fish", " air", " ship"; " which" nearest " who"; "er" nearest
  "ers", "ering"), and near-miss firing still only hurts.
- The probability-space gain beyond the controls grows with the
  vocabulary: about 0.03 nats on corpus20, 0.04 at V = 512 here, 0.11 at
  V = 2048. At V = 2048 neither the mix nor the unigram control had
  levelled off at the largest weight tried (0.2 and 0.1).
- Top-1 accuracy moves by at most 0.2 points in any row that helps.

### Wide-window head (measured, not adopted)

The idea: one more head averages the tokens further back than the chain
sees (distances 4 to W from the token being predicted), and a direct linear
map from that average adds to the logits. The map is a table of log ratios
counted from the training text: how often token v followed a window holding
token u, over how often it was expected, with k pseudo-counts on both. Two
versions of "expected": the unigram rate, and the chain's own prediction
with each half of the training text scored by a chain built on the other
half. Measured on the chain's exact logits, split 0, all held-out positions,
with the scale picked on the held-out text (so gains are upper bounds).
Nothing in `src/` changed.

| | corpus20 | Moby Dick V = 512 | Moby Dick V = 2048 |
|---|---|---|---|
| chain | 2.643 | 3.057 | 4.603 |
| trigger table against unigram, W = 50 (best scale) | 2.628 | 3.046 | 4.576 |
| the same table, window taken from elsewhere in the text | 2.642 | 3.057 | 4.603 |
| trigger table against the chain, W = 50 | 2.634 | 3.052 | 4.587 |
| logit bonus for tokens present in the window, W = 50 | 2.642 | 3.054 | 4.595 |
| cache: mix in the window's token frequencies, probability space, W = 1000 | 2.614 (W = 200) | 3.032 | 4.546 |
| control: unigram mix | 2.606 | 3.032 | 4.522 |
| control: logits divided by temperature 1.2 | 2.574 | 2.974 | 4.465 |

- **The window carries real signal, and it is small.** The best gain is
  0.011 to 0.014 nats at V = 512 and about 0.03 at V = 2048. A window from
  elsewhere in the text gains nothing, so it is content and not hedging.
  Top-1 accuracy moves by at most 0.2 points.
- W = 50 is the best window; 200 and 1000 are worse for the trigger table.
  The unigram version beats the cross-fitted chain version everywhere.
- The strongest learned triggers are sensible at V = 2048 (" we" raises
  " our", " you" raises " your", " more" raises " than", and most common
  words raise themselves by about 0.4 nats) and mostly punctuation and
  byte fragments at V = 512.
- The probability-space cache does no better than mixing in plain unigram.
- **Plain temperature beats everything tried so far**: dividing the chain's
  logits by 1.2 gains 0.07 nats on corpus20, 0.08 and 0.14 on Moby Dick.
  The smoothed chain is overconfident, as the unigram control already
  said. This compiles as a rescaled decoder and output bias, with no new
  parts. It was measured on one split with the temperature picked on the
  held-out text; not yet built or checked across splits.

### Moby Dick compile (measured, `moby_dick.toml`)

Squeezed build, V = 512, context 3, prune 0.5, split 0: 512,011 training
and 57,072 held-out tokens; 135,713 contexts kept of 196,170 (pruning costs
KL 0.028 against the unpruned chain on the training text). `d_model` 224,
`d_ff` 135,713, 61 M weights (30 M nonzero). The whole run takes 8 min 25 s
on 24 CPU threads, with a scratch job running beside it.

| | training text | held-out |
|---|---|---|
| mean KL against the chain (nats) | 0.024 | 0.023 |
| top-1 agreement | 98.7% | 98.9% |
| NLL: chain / transformer | 1.476 / 1.498 | 3.060 / 3.000 |

- Rank 128 holds up with 7.6 times the contexts: KL doubles from corpus20's
  0.012 and top-1 agreement is unchanged.
- Held-out, mean of 3 splits: unigram 5.24, 1-token chain 3.76 (13.9%),
  unpruned 3.069 (33.9%), as configured 3.063 (33.7%). Splits differ by
  0.03 nats here, against more than a nat on corpus20. Pruning slightly
  helps held out.
- Coupled samples leave the chain's within about ten tokens (2 to 11 of 80
  identical), far sooner than on corpus20, since one differing draw is
  enough. Both read as local Moby Dick with no thread across a sentence.
- `hegel_generate.py` on this config has not been timed.

### Hegel build (measured, `hegel.toml`)

`resources/hegel.txt` is 13.1 M characters of Hegel in English, several
works run together, from OCR: footnote debris and German fragments are in
it and show up in samples. Same settings as Moby Dick.

`just build hegel`, whole corpus: 5,682,998 tokens, 391,236 contexts kept,
`d_model` 224, `d_ff` 391,236, 176 M weights (87 M nonzero). It takes 7
minutes on 24 CPU threads and peaks at 9.6 GB of memory.
`compiled/hegel.pt` is 704 MB. `just chat hegel` is ready in about 20 s,
and an 80-token answer takes a few seconds.

- A hundred times corpus20's text gives only 22 times the contexts.
- `just compile hegel` (the held-out check) takes 2 h 20 min and peaks at
  about 14 GB, because the model runs over all the training text at 1 to
  2 s per 1000 tokens. Checking a sample of the training text would fix
  that; not built.

`just compile hegel`, split 0: 5,115,014 training and 568,058 held-out
tokens; 372,721 contexts kept of 521,923 (pruning costs KL 0.006 against
the unpruned chain on the training text); `d_ff` 372,721, 168 M weights.

| | training text | held-out |
|---|---|---|
| mean KL against the chain (nats) | 0.033 | 0.033 |
| top-1 agreement | 98.9% | 98.9% |
| NLL: chain / transformer | 1.819 / 1.858 | 2.426 / 2.389 |

- Rank 128 still holds with 21 times corpus20's contexts: KL 0.033
  (corpus20 0.012, Moby Dick 0.024), top-1 agreement unchanged.
- Held-out, one split: unigram 5.26, 1-token chain 3.49 (16.6%), unpruned
  2.428 (43.1%), as configured 2.426 (43.1%). Hegel is much more
  predictable than Moby Dick (3.06, 33.7%) at the same settings.
- Coupled samples leave the chain's within 3 to 7 tokens of 80.

### Held-out behaviour (measured)

Numbers are NLL in nats per token; see the table in "Status".

- **Hard backoff is badly calibrated, not badly informed.** It scores 5.29
  held out, about as bad as unigram, yet its top-1 accuracy (48.6%) equals
  the smoothed chain's. A context seen once predicts its one continuation
  with near certainty; about a third of held-out tokens land on the floor at
  15.4 nats each, and that alone accounts for nearly the whole score.
- Under hard backoff, longer contexts made it worse (context 1: 3.99,
  2: 5.35, 3: 5.91, 4: 6.08 on split 0), a bigger floor helped (floor 0.1:
  3.59) and so did pruning. The best that design reached was about 3.17.
- **Smoothing fixes it**: 2.45 unpruned, and longer contexts help again up
  to 3.
- **One split is a noisy measure.** Hard backoff by split: random seeds 0 to
  5 give 5.88, 5.30, 4.65, 5.31, 5.33, 5.15; the last 10% gives 4.10, the
  first 10% 5.84, the middle 10% 7.10. Compare methods on the same seeds.
- **The squeezed transformer scores a little better than its own chain**
  held out (2.620 against 2.662; under hard backoff 5.57 against 5.88). This
  is not generalisation: its approximation error spreads a little
  probability around, which is a softer hedge.

### Verification (implemented)

- `FidelityMeter`: mean and max KL (reference first), top-1 agreement, largest
  log-probability difference, and both predictors' NLL on the actual next
  token. Reference tokens within 1e-4 nats of the maximum count as tied, so
  rounding an exact tie is not a disagreement.
- The stream is fed in chunks with `max_offset` tokens of lead-in, which
  equals one forward pass (tested).
- Samples from chain and model use the same uniform draws through the inverse
  CDF, so they stay in step exactly as long as the distributions agree.
- `FidelityMeter` also reports each predictor's top-1 accuracy on the actual
  next token.
- With a held-out split, `hegel_compile.py` prints held-out NLL and top-1
  accuracy for unigram, a 1-token chain, the unpruned chain and the chain
  over `holdout_seeds` splits, then the transformer against its chain on the
  first split.

## Open questions

Tracked in `PRODUCT.md`: shrinking `d_ff`, the closeness threshold,
vocabulary size and context length, whether token similarity pays, and what
would give samples a thread longer than the context (the wide-window head
did not).
