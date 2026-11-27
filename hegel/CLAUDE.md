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

Next step: structured (not random) token vectors, so a near-miss context
fires usefully instead of being suppressed. Now that the chain is
calibrated, a held-out improvement from that would be visible.

## Files and commands

```
hegel_compile.py        entry point: tokenize, chain, compile, verify, sample
hegel_generate.py       source string in, continuation out; or interactive
resources/corpus20.toml           the exact wide build
resources/corpus20_squeezed.toml  32-dim slots, 128-dim logit code
src/config.py           HegelConfig, load_config (TOML, unknown keys rejected)
src/corpus.py           normalize (one stream), tokenize (BPE from semantic/)
src/chain.py            count_contexts, build_chain (pruning), Chain (backoff)
src/model.py            HegelModel: the minimal transformer, all weights zero
src/factor.py           random codes; weighted low-rank factoring of the table
src/compiler.py         compile_chain (wide or squeezed), compile_wide
src/pipeline.py         compile_from_config: corpus -> tokens -> chain -> model
src/holdout.py          score the chain on several random held-out splits
src/verify.py           FidelityMeter, chain/model comparisons, coupled sampling
src/generation.py       respond: continue a source string
tests/                  one test file per src module
```

```bash
just hegel                           # compile + verify corpus20 (wide)
just hegel corpus20_squeezed         # the squeezed build
just hegel-run "source string"       # continue a string (wide build)
just hegel-run "source string" corpus20_squeezed
just hegel-run                       # interactive; empty line quits
cd hegel
uv run python hegel_compile.py resources/corpus20_squeezed.toml
uv run python hegel_generate.py resources/corpus20.toml "source string"
uv run pytest tests        # 101 tests, about 15 s
uv run black src tests hegel_compile.py hegel_generate.py
uv run isort src tests hegel_compile.py hegel_generate.py
uv run flake8 --max-line-length 88 --extend-ignore E203 src tests *.py
uv run mypy src hegel_compile.py hegel_generate.py
```

Run from `hegel/`: scripts and tests import `src.*` relative to it.

`hegel_generate.py` recompiles from the config on every start (about 15 s
wide, 40 s squeezed, most of it the output factoring) instead of loading a
checkpoint, so there is nothing to keep in sync. Interactive mode pays that
once. It sets `holdout_fraction` to 0, so the model you talk to is built on
the whole corpus.

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

Not tried yet: structured (not random) token vectors, which is where
near-miss firing could become generalisation instead of noise.

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

Tracked in `PRODUCT.md`: structured token vectors, shrinking `d_ff`, the
closeness threshold, vocabulary size and context length.
