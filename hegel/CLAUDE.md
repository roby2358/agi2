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

Two builds run end to end on `../samples/corpus20.txt`, on a Kneser-Ney
smoothed chain (discount 0.8). A random 10% of the text is held out: 49,416
training tokens, 5,540 held-out tokens, V = 512, context 3, prune threshold
0.5, 11,393 contexts kept of 31,873. Fidelity is the transformer against
the chain it was compiled from.

| | wide (`corpus20`) | squeezed (`corpus20_squeezed`) |
|---|---|---|
| slots / logit region | one-hot 512 / exact 512 | random 32-dim / 128-dim code |
| `d_model`, `d_ff` | 2048, 11,393 | 224, 11,393 |
| weights | 53 M (5.9 M nonzero) | 5.4 M (2.5 M nonzero) |
| training text: mean KL (nats) | below 1e-6 | 0.013 |
| training text: top-1 agreement | 100% | 99.5% |
| training text: NLL (chain 1.345) | 1.345 | 1.376 |
| held-out: mean KL (nats) | below 1e-6 | 0.013 |
| held-out: top-1 agreement | 100% | 99.4% |
| held-out: NLL (chain 2.529) | 2.529 | 2.518 |

The chain as a language model, held-out NLL in nats per token, mean over 5
random splits with the range in brackets:

| predictor | NLL | top-1 accuracy |
|---|---|---|
| uniform | 6.24 | |
| unigram | 5.32 | |
| chain, 1 token of context | 3.43 [3.35, 3.50] | 17.3% |
| hard backoff, context 3 (`smoothing = "none"`) | 5.29 [4.65, 5.88] | 48.6% |
| Witten-Bell, context 3, unpruned | 2.45 [2.21, 2.66] | 48.8% |
| Witten-Bell, context 3, prune 0.5 (17,915 contexts) | 2.47 [2.23, 2.66] | 47.7% |
| Kneser-Ney, context 3, unpruned | 2.31 [2.12, 2.48] | 48.6% |
| Kneser-Ney, context 3, as configured (11,370 contexts) | 2.37 [2.20, 2.53] | 46.0% |

`hegel_generate.py` continues a source string with either build, compiled
on the whole corpus.

The squeezed build also compiles on `../samples/moby_dick_pg2701.txt` (ten
times the text) and on `resources/hegel.txt` (a hundred times); see "Moby
Dick compile" and "Hegel build" below. The Hegel weights are saved and can
be used without recompiling.

Kneser-Ney replaced Witten-Bell in all four configs: on every corpus it
scores better held out with about half the contexts, and the squeezed
transformer follows it at least as closely. It also makes context 4 and
longer pay, which Witten-Bell did not (see "Smoothing for longer context").
Context 4 runs through the chain builder and the squeezed compile on Hegel:
2.298 held out against 2.347, for twice the contexts and 2.3 times the
weights, with the transformer nearly as close to its chain (see "Context 4
on Hegel"). Context 3 is still what ships; no config sets 4.

What comes next is in the backlog, `pjpd/` in this directory (the
`pjpd_*` tools, with Claude Code opened in `hegel/`; see the
`pjpd-process` skill). Tasks are ready to work; ideas need a conversation
with the user first. At the top: the looped block on the wide build. Do
not keep a to-do list here.

Two extensions were measured as scratch scripts and not adopted: structured
token vectors and a wide-window head (sections below). Both were measured
on the Witten-Bell chain, as was the temperature 1.2 result; none has been
rechecked under Kneser-Ney.

## Files and commands

```
hegel_compile.py        entry point: tokenize, chain, compile, verify, sample
hegel_build.py          compile on the whole corpus and save the weights
hegel_generate.py       source string in, continuation out; or interactive
resources/corpus20.toml           the exact wide build
resources/corpus20_squeezed.toml  32-dim slots, 128-dim logit code
resources/moby_dick.toml          the squeezed build on Moby Dick (about 3 min)
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
justfile                the compile, build, chat and run recipes, and cc-*
runs/                   logs of timed runs (just cc-timed); not in git
pjpd/                   the backlog: tasks.txt, ideas.txt (pjpd tools)
../.claude/skills/hegel-lab/   the skill behind the cc- recipes, with scripts
```

```bash
cd hegel
just compile                         # compile + verify corpus20 (wide)
just compile corpus20_squeezed       # the squeezed build
just compile moby_dick               # squeezed, Moby Dick, about 3 min
just run "source string"             # continue a string (wide build)
just run "source string" corpus20_squeezed
just run                             # interactive; empty line quits
just build hegel                     # whole corpus, save compiled/hegel.pt, 5 min
just chat hegel                      # interactive, from the saved weights
uv run python hegel_compile.py resources/corpus20_squeezed.toml
uv run python hegel_generate.py resources/corpus20.toml "source string"
uv run pytest tests        # 111 tests, about 15 s
just cc-check              # black, isort, flake8, mypy, pytest; changes nothing
just cc-fmt                # black and isort, rewriting files
```

Run from `hegel/`: scripts and tests import `src.*` relative to it.

The `cc-` recipes script what kept being retyped: `cc-check`, `cc-fmt`,
`cc-timed` and `cc-status` (long runs with a log, time and peak memory),
`cc-stats` (size up a corpus), `cc-sweep` and `cc-squeeze` (one setting,
several values), `cc-context` and `cc-smoothing` (context lengths past 3), `cc-throughput`, `cc-py` (run a scratch script),
`cc-commit` (stage all, commit from a message file, push). The
`hegel-lab` skill says when to use each. Use them rather than typing the
commands out; a measurement run twice becomes a script there with a recipe.

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
  predicts alone. With `smoothing = "kneser_ney"` (the shipped setting) a
  context takes `discount` off each of its counts and gives the freed mass,
  discount * t / n for n counted and t distinct continuations, to the
  distribution of the context one token shorter, recursively down to the
  floored unigram. Contexts of full length count occurrences. Shorter ones
  count, for each next token, how many different tokens preceded the
  context when it followed, so a continuation that only ever comes after
  one longer context is not credited to the shorter one again. The mix
  always uses the next context down, whether or not pruning keeps it.
  `smoothing = "witten_bell"` mixes plain relative frequencies with own
  share n / (n + t). `smoothing = "none"` is hard backoff: each context's
  own frequencies and the floor.
- **Discount.** 0.8 on corpus20, 0.9 on Moby Dick and Hegel, picked on
  held-out text. A larger discount also prunes harder at the same
  threshold. One discount for every count; a discount per count (modified
  Kneser-Ney) is not tried.
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

### Pruning (measured, mean of 5 splits)

Kneser-Ney, discount 0.8 (`just cc-sweep corpus20 prune_threshold ...`):

| threshold (nats) | contexts kept | held-out NLL | held-out top-1 |
|---|---|---|---|
| 0 | 32,139 | 2.309 | 48.6% |
| 0.1 | 21,017 | 2.325 | 48.0% |
| 0.2 | 16,649 | 2.339 | 47.3% |
| 0.5 (shipped) | 11,370 | 2.369 | 46.0% |
| 1 | 8,355 | 2.405 | 44.9% |

- Pruning costs more held out than it did under Witten-Bell (0.06 nats at
  0.5 against 0.013), and keeps far fewer contexts at the same threshold.
  On split 0 the training text goes from NLL 1.035 unpruned to 1.345, with
  the top token changed at 14% of positions. Threshold 0.2 gives about the
  old model size and 2.339; 0.5 stays for now, unexamined.
- On the larger corpora pruning is nearly free: Moby Dick 2.879 unpruned,
  2.900 at 0.5; Hegel 2.341 and 2.347.
- By discount at threshold 0.5: 0.6 gives 2.352 (16,506 contexts), 0.7
  2.343 (14,208), 0.8 2.369 (11,370), 0.9 2.435 (8,433). Unpruned, 0.7 and
  0.8 tie at 2.31.

Witten-Bell, for the record:

| threshold (nats) | contexts kept | held-out NLL | held-out top-1 | training NLL |
|---|---|---|---|---|
| 0 | 32,139 | 2.453 | 48.8% | 0.745 |
| 0.5 | 17,915 | 2.466 | 47.7% | 0.894 |
| 2 | 7,774 | 2.516 | 43.7% | 1.337 |
| 8 | 2,397 | 2.691 | 37.0% | 1.998 |

- Under Witten-Bell context 4 did not help (2.46 at best); context 2 gave
  2.55.
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

Kneser-Ney chain, full model on split 0 (slots 32, tau 0.7, 4 rounds):
rank 128 gives training KL 0.013 and 99.5% top-1, held-out KL 0.013 and
99.4%. Under Witten-Bell it was 0.012 with 99.0% and 98.5%; from the
scratch runs on that factoring alone, rank 64 gave KL 0.076 and 97.6%,
rank 192 about 0.005. The table below is also from Witten-Bell ("smoothed")
and hard backoff.

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

Squeezed build, V = 512, context 3, Kneser-Ney discount 0.9, prune 0.5,
split 0: 512,011 training and 57,072 held-out tokens; 47,062 contexts kept
of 196,170 (pruning costs KL 0.027 against the unpruned chain on the
training text, NLL 1.905 to 2.236). `d_model` 224, `d_ff` 47,062, 21 M
weights (10 M nonzero). The whole run takes 3 min 20 s on 24 CPU threads
and peaks at 3.8 GB.

| | training text | held-out |
|---|---|---|
| mean KL against the chain (nats) | 0.016 | 0.015 |
| top-1 agreement | 99.3% | 99.3% |
| NLL: chain / transformer | 2.236 / 2.280 | 2.893 / 2.878 |

- Held-out, mean of 3 splits: unigram 5.24, 1-token chain 3.75 (13.9%),
  unpruned 2.879 (34.1%), as configured 2.900 (33.6%). Splits differ by
  0.03 nats here, against more than a nat on corpus20.
- Under Witten-Bell the same config kept 135,713 contexts (61 M weights,
  8 min 25 s) and scored 3.063 held out, with KL 0.024 and 98.7% top-1
  against its chain. Kneser-Ney is a third the size, 0.16 nats better and
  closer to its chain.
- Discount 0.8 keeps 66,756 contexts for 2.895; 0.9 was taken for size.
- Coupled samples leave the chain's within about ten tokens (2 to 11 of 80
  identical), far sooner than on corpus20, since one differing draw is
  enough. Both read as local Moby Dick with no thread across a sentence.
- `hegel_generate.py` on this config has not been timed.

### Hegel build (measured, `hegel.toml`)

`resources/hegel.txt` is 13.1 M characters of Hegel in English, several
works run together, from OCR: footnote debris and German fragments are in
it and show up in samples. Same settings as Moby Dick.

Kneser-Ney, discount 0.9.

`just build hegel`, whole corpus: 5,682,998 tokens, 207,974 contexts kept,
`d_model` 224, `d_ff` 207,974, 94 M weights (46 M nonzero). It takes 5
minutes on 24 CPU threads and peaks at 6.6 GB of memory.
`compiled/hegel.pt` is 375 MB. A forward pass over 1024 tokens takes
about 0.9 s.

Split 0 (5,115,014 training and 568,058 held-out tokens): 195,534 contexts
kept of 521,923. Chain held out: unigram 5.26, 1-token chain 3.49 (16.6%),
unpruned 2.341, as configured 2.347 (43.1%). Prune 0.2 keeps 243,271 for
2.345; prune 1 keeps 165,551 for 2.349.

Transformer against its chain, from `just cc-squeeze hegel logit_dim 128`
(the first 20,000 positions of each text, 5 min, peak 6.2 GB):

| | training text | held-out |
|---|---|---|
| mean KL against the chain (nats) | 0.017 | 0.017 |
| top-1 agreement | 99.5% | 99.4% |
| NLL: chain / transformer | 2.061 / 2.098 | 2.313 / 2.303 |

- The full check, `just compile hegel`, has not been rerun under
  Kneser-Ney. Under Witten-Bell it took 2 h 20 min and peaked at about
  14 GB, because the model runs over all the training text.
- Under Witten-Bell: 372,721 contexts on the split (391,236 on the whole
  corpus, 176 M weights, 704 MB, 7 min, 9.6 GB), chain 2.426 held out,
  transformer KL 0.033 and 98.9% top-1 over all positions. Kneser-Ney is
  half the size, 0.08 nats better and closer to its chain.
- Hegel is much more predictable than Moby Dick (2.90, 33.6%) at the same
  settings.

### Longer context on Hegel (measured, `just cc-context hegel 6`)

Unpruned smoothed chain, split 0, all 568,058 held-out tokens. The scorer
is `long_context.py` in the hegel-lab skill; it reproduces the chain's
numbers at context 1 and 3 exactly.

| context | distinct contexts of that length | seen twice or more | held-out NLL | top-1 | held-out positions matching at full length |
|---|---|---|---|---|---|
| 1 | 364 | 363 | 3.493 | 16.6% | 100% |
| 2 | 39,577 | 33,789 | 2.595 | 36.6% | 99.9% |
| 3 | 481,982 | 273,162 | 2.428 | 43.1% | 95.7% |
| 4 | 1,525,429 | 535,728 | 2.483 | 44.8% | 80.2% |
| 5 | 2,647,755 | 586,351 | 2.575 | 45.4% | 59.1% |
| 6 | 3,547,777 | 510,876 | 2.657 | 45.6% | 40.1% |

- **Longer context knows more and scores worse.** Top-1 accuracy rises 2.5
  points from context 3 to 6 while NLL gets worse by 0.23 nats. The same
  on corpus20 (context 4: 2.707 against 2.659, top-1 46.3% against 45.3%).
  That is calibration, not information: Witten-Bell gives a context seen
  once with one continuation half the say, and most long contexts are
  seen once.
- So context 3 is the best Witten-Bell can use. Better smoothing fixes
  it; see the next section.
- Past context 3 the contexts multiply faster than they repeat: length 4
  has 3.2 times the distinct contexts of length 3 and 2.0 times those
  seen twice.

### Smoothing for longer context (measured, `just cc-smoothing hegel 6`)

Same split and scorer; held-out NLL with top-1 accuracy in brackets, best
setting tried of each kind. `wb k` is Witten-Bell with own share
n / (n + k t). `ad D` is interpolated absolute discounting on raw counts.
`kn D` is Kneser-Ney: the same, with lower levels counting distinct left
neighbours instead of occurrences.

| smoothing | context 3 | 4 | 5 | 6 |
|---|---|---|---|---|
| wb 1 (shipped) | 2.428 (43.1%) | 2.483 (44.8%) | 2.575 (45.4%) | 2.657 (45.6%) |
| wb 3 | 2.363 (43.0%) | 2.324 (45.0%) | 2.329 (45.8%) | 2.345 (46.1%) |
| wb 8 | 2.389 (42.3%) | 2.312 (44.5%) | 2.283 (45.6%) | 2.272 (46.1%) |
| ad 0.9 | 2.423 (43.1%) | 2.424 (45.0%) | 2.447 (45.8%) | 2.468 (46.2%) |
| kn 0.8 | 2.343 (43.2%) | 2.298 (45.0%) | 2.292 (45.8%) | 2.297 (46.1%) |
| kn 0.9 | 2.341 (43.2%) | 2.285 (45.1%) | 2.266 (46.0%) | 2.259 (46.4%) |

- **Kneser-Ney makes longer context pay**: 2.259 at context 6 against the
  shipped 2.428, and 3.3 points of top-1. Most of it comes by context 4
  (2.285); 5 and 6 add 0.02 and 0.01.
- **It also helps at context 3 with no more contexts**: 2.341, a gain of
  0.087 nats, more than temperature 1.2 gave anywhere.
- Scaling Witten-Bell's backoff share gets most of the way, but the best
  k grows with the context (3 at context 3, 8 at context 6).
- On corpus20 split 0 the same holds: kn 0.8 gives 2.477 at context 3 and
  2.446 at context 4, against 2.659 shipped.
- D = 0.9 is the edge of the grid and still the best; one discount for
  every count is crude, and a discount per count (modified Kneser-Ney)
  was not tried. The settings were picked on the held-out text.
- Adopted at context 3 (`smoothing = "kneser_ney"` in `src/chain.py`,
  which reproduces this scorer's numbers exactly). Context 4 is measured
  through `build_chain` and the squeezed compile in the next section.
  Context 5 and longer are not built.

### Context 4 on Hegel (measured, not shipped)

`hegel.toml` with `max_context` overridden to 4 and nothing else changed
(Kneser-Ney 0.9, prune 0.5, split 0, 5,115,014 training and 568,058
held-out tokens). No config sets it.

The chain, from `just cc-sweep hegel max_context 3 4` (4.7 min in all,
peak 18.8 GB; context 4 alone 234 s):

| context | contexts kept | held-out NLL | top-1 | unpruned NLL |
|---|---|---|---|---|
| 3 | 195,534 | 2.347 | 43.1% | 2.341 |
| 4 | 398,604 | 2.298 | 44.8% | 2.285 |

Contexts at context 4, by length:

| length | distinct | seen twice or more | kept |
|---|---|---|---|
| 1 | 364 | 363 | 363 |
| 2 | 39,577 | 33,789 | 26,510 |
| 3 | 481,982 | 273,162 | 119,383 |
| 4 | 1,525,429 | 535,728 | 252,348 |

The squeezed transformer against its chain, from `just cc-squeeze hegel
max_context 3 4` (slots 32, rank 128, tau 0.7, 4 rounds; the first 20,000
positions of each text; 11.8 min for both, peak 12.0 GB). Context 3 took
29 s to build the chain and 215 s to compile, context 4 took 62 s and
323 s. The context 3 column repeats the table under "Hegel build".

| | context 3 | context 4 |
|---|---|---|
| `d_model`, heads | 224, 2 | 256, 3 |
| `d_ff` | 195,534 | 398,604 |
| weights | 88.1 M (43.2 M nonzero) | 204.8 M (97.0 M nonzero) |
| training text: mean KL (nats) | 0.017 | 0.018 |
| training text: top-1 agreement | 99.5% | 99.2% |
| training text: NLL chain / transformer | 2.061 / 2.098 | 1.865 / 1.907 |
| held-out: mean KL (nats) | 0.017 | 0.017 |
| held-out: top-1 agreement | 99.4% | 99.2% |
| held-out: NLL chain / transformer | 2.313 / 2.303 | 2.272 / 2.257 |

- **The real builder agrees with the lean scorer**: 2.285 unpruned at
  context 4, the same as `cc-smoothing`.
- **Context 4 pays after pruning too**: 0.049 nats and 1.7 points of
  top-1 over context 3. Pruning costs 0.013 nats at context 4 against
  0.006 at context 3.
- **It costs twice the neurons**: 398,604 against 195,534, and 2.3 times
  the weights because `d_model` grows by one slot. Pruning keeps 17% of
  the length-4 contexts.
- Fewer length-3 contexts are kept than at context 3: 119,383 against
  168,661, with lengths 1 and 2 unchanged (146,256 of length 1 to 3 in
  all, against 195,534). The likely reason, not checked: length-3 contexts
  now count distinct left neighbours, not occurrences, so their rows
  differ from context 3's.
- **Rank 128 still fits the rows.** KL is unchanged at 0.017 to 0.018;
  top-1 agreement drops 0.2 to 0.3 points.
- **Memory is the limit.** Measured: the squeeze run peaks at 12.0 GB
  (context 3 alone 6.2 GB) and the sweep at 18.8 GB of the 20.5 GB on
  this machine (context 3 alone 8.4 GB). The sweep's extra is put down to
  the unpruned chain it also scores; that is read from the code, not
  profiled. Unpruned, every
  context is kept: 2.05 M rows of 4 KB is 8.4 GB, which `build_chain`
  holds as a list and then copies into one array. By the same arithmetic
  context 5 unpruned (4.7 M rows, 19 GB) will not fit; the pruned build
  might. Neither was tried.
- corpus20 shows the same pattern (`just cc-squeeze corpus20_squeezed
  max_context 3 4`, split 0): 17,734 contexts kept against 11,393, the
  length-3 ones falling from 6,755 to 4,661; KL 0.013 at both, top-1
  agreement 99.4% / 99.3% against 99.4% / 99.4%; held-out NLL 2.495
  against 2.529 for the chain on that one split.
- Not done at context 4: the whole-corpus build and saved weights, the
  full check over every position, samples, and any retuning of the
  discount, the prune threshold or the rank.

### Looped block (design, not built)

One block with shared weights, applied n - 1 times for context n, with two
heads whatever n is. The current block cannot simply be repeated: its
heads add tokens into slots and its neurons add to the logits, so a second
pass doubles both.

- Head A looks one token back and copies that position's slots shifted by
  one: its slot 0 into my slot 1, its slot 1 into my slot 2, and so on.
- Head B attends to the position itself and subtracts its old slots 1 and
  up and its old logit region, so each pass sets them instead of adding.
- After pass t, slots 1 to t are right and stay right. The context
  neurons rewrite the logits from the slots on every pass, so the last
  pass equals the chain. No step counter is needed.
- Check it on the wide build, where the match should be exact. In the
  squeezed build the slots are approximate and the error is fed back each
  pass; that is the risk.
- It saves heads, not neurons: `d_ff` is still one per kept context, and
  each pass costs a full forward pass. A block with distinct weights per
  layer would be the easier compile; the loop's point is depth chosen at
  run time.

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
