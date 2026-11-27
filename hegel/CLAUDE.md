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

Two builds run end to end on `../samples/corpus20.txt`. A random 10% of the
text is held out: 49,416 training tokens, 5,540 held-out tokens, V = 512,
context 3, prune threshold 0.5, 22,611 contexts kept of 31,873. Fidelity is
the transformer against the chain it was compiled from.

| | wide (`corpus20`) | squeezed (`corpus20_squeezed`) |
|---|---|---|
| slots / logit region | one-hot 512 / exact 512 | random 32-dim / 128-dim code |
| `d_model`, `d_ff` | 2048, 22,611 | 224, 22,611 |
| weights | 99 M (626 K nonzero) | 10.4 M (4.9 M nonzero) |
| training text: mean KL (nats) | 3.1e-7 | 0.0022 |
| training text: top-1 agreement | 100% | 99.78% |
| training text: NLL (chain 0.5112) | 0.5112 | 0.5137 |
| held-out: mean KL (nats) | 2.1e-7 | 0.016 |
| held-out: top-1 agreement | 100% | 97.9% |
| held-out: NLL (chain 5.883) | 5.883 | 5.570 |

The compile works: both builds follow their chain on text the chain never
saw, not only on the training text. **The chain itself is now the weak
link.** On held-out text it scores 5.88 nats per token, worse than unigram
frequencies (5.32) and far worse than a chain with one token of context
(3.99). See "Held-out behaviour" below.

`hegel_generate.py` continues a source string with either build.

Next step, proposed and not started: smooth the chain (interpolate each
context with the one it backs off to). In a scratch test this takes held-out
NLL from 5.88 to 2.66.

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
uv run pytest tests        # 84 tests, about 15 s
uv run black src tests hegel_compile.py hegel_generate.py
uv run isort src tests hegel_compile.py hegel_generate.py
uv run flake8 --max-line-length 88 --extend-ignore E203 src tests *.py
uv run mypy src hegel_compile.py hegel_generate.py
```

Run from `hegel/`: scripts and tests import `src.*` relative to it.

`hegel_generate.py` recompiles from the config on every start (about 15 s
wide, 45 s squeezed, nearly all of it the output factoring) instead of
loading a checkpoint, so there is nothing to keep in sync. Interactive mode
pays that once.

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
- **Longest-match backoff.** The longest kept context that matches predicts;
  shorter ones are fallback, down to the unigram distribution. The original
  chain sums counts across context lengths, which a softmax output cannot
  express directly. Summing may be revisited.
- **Single-token suffixes.** A context maps to one next token.
- **Context length counts the current token.** `max_context = 3` means the
  current token and the two before it, so 2 attention heads.
- **Floor.** Every distribution is `(1 - floor) * relative frequency +
  floor * uniform`, with `floor = 1e-4`.
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

### Pruning (measured on corpus20, V = 512, context 3)

The tables in this section and in "Squeezed build" were measured with the
chain built on the whole corpus, before the held-out split existed. With 10%
held out the numbers move slightly (see "Status").

Pruned chain against the unpruned chain, over all corpus positions:

| threshold (nats) | contexts kept | mean KL | top-1 agreement |
|---|---|---|---|
| 0 | 34,075 | 0 | 100% |
| 0.5 | 24,419 | 0.010 | 99.8% |
| 1 | 20,903 | 0.057 | 98.1% |
| 2 | 15,251 | 0.210 | 91.0% |
| 4 | 8,100 | 0.580 | 77.1% |
| 8 | 4,082 | 0.983 | 63.7% |
| 16 | 2,134 | 1.376 | 52.1% |

There is no knee; cost rises steadily. 0.5 is the default because it removes
28% of the contexts for 0.01 nats. These numbers are in-sample: the unpruned
chain scores 0.53 nats per token on its own corpus, which is mostly recall.

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
a `logit_dim` code and the decoder has one vector per token. Each context
wants `log P - floor` on its seen tokens and 0 on the rest, with weight 1 on
seen tokens and `unseen_weight` (0.01) on unseen ones. Codes and decoder are
fitted by alternating closed-form solves, starting from a random decoder.
Contexts are weighted by how often they predict, plus 1. Neurons write code
deltas against their parent exactly as in the wide build, and telescoping
still holds because it is linear.

What was tried, at rank 128 unless noted (mean KL in nats, exact context
detection):

| method | KL | top-1 |
|---|---|---|
| SVD of the log-probability table (any centring or row weighting) | 1.9 | 72% |
| SVD at rank 256 | 0.3 | 96% |
| random decoder, plain projection | 0.49 | 84% |
| random decoder, weighted solve per context (`decoder_rounds = 0`) | 0.018 | 98.8% |
| the same, with a larger seen/unseen margin | worse | |
| alternating least squares, 4 rounds | 0.002 | 99.8% |

- SVD fails because the rows are nearly one-hot: an even-handed fit in log
  space spends its rank on floor entries.
- With a random decoder, contexts with 1 to 4 possible next tokens (84% of
  positions) are near-perfect, but broad ones leak: rows with 17 or more
  seen tokens put 39% of their mass on tokens the chain never allows, and
  the unigram row is destroyed (KL 8.4), which shows up as byte garbage in
  samples. Refitting the decoder removes that: leak falls to 0.05% in every
  breadth band.
- The `+ 1` in the row weights matters for the unigram row, which never
  predicts on the corpus but is what an off-corpus prompt falls back to.

Full model, slots 32, tau 0.7, 4 rounds (`unigram KL` is the unigram row
alone):

| `logit_dim` | `d_model` | mean KL | top-1 | unigram KL | factoring time |
|---|---|---|---|---|---|
| 32 | 128 | 0.200 | 88.8% | 2.0 | 1 s |
| 48 | 144 | 0.050 | 95.6% | 1.5 | 6 s |
| 64 | 160 | 0.020 | 98.3% | 1.3 | 9 s |
| 96 | 192 | 0.0044 | 99.5% | 0.51 | 19 s |
| 128 | 224 | 0.0022 | 99.8% | 0.08 | 32 s |
| 192 | 288 | 0.0007 | 99.9% | 0.004 | 71 s |

More rounds help low ranks a little (rank 64, 8 rounds: KL 0.0175) and mostly
repair the unigram row. 128 is the default because it is the smallest rank
where the unigram row is also good.

Not tried yet: structured (not random) token vectors, which is where
near-miss firing could become generalisation instead of noise.

### Held-out behaviour (measured; the chain is the problem)

NLL in nats per token on the held-out 10%, `holdout_seed = 0` (uniform is
6.24, unigram 5.32):

- **The configured chain scores 5.88**, worse than unigram. 34% of held-out
  tokens land on the floor at 15.4 nats each. A context seen once predicts
  its one continuation with probability near 1; on new text that is often
  wrong, and hard backoff has no way to hedge.
- **Longer contexts make it worse.** Unpruned, floor 1e-4: context 1 scores
  3.99, context 2 5.35, context 3 5.91, context 4 6.08. In-sample the order
  is reversed (3.03, 1.17, 0.50, 0.28).
- **The floor is the biggest lever within the current design.** Context 3,
  unpruned: floor 1e-4 gives 5.91, 1e-3 5.11, 1e-2 4.32, 3e-2 3.96, 1e-1
  3.59.
- **Pruning helps out of sample**, the opposite of in-sample. Context 3,
  floor 1e-4: threshold 0.5 gives 5.88 (22,611 contexts), 4 gives 5.25
  (7,456), 16 gives 4.25 (1,946), 32 gives 3.92 (958).
- **The best this design reaches is about 3.17** (floor 0.1, threshold 16,
  about 1,700 contexts). A uniform floor is a crude hedge.
- **Interpolated smoothing reaches 2.66.** Scratch only: each context's
  distribution is mixed with its parent's, unpruned. Witten-Bell weights
  (no tunable): context 1 3.51, context 2 2.71, context 3 2.66, context 4
  2.71. Absolute discounting at 0.75: 3.52, 2.74, 2.65, 2.66. Longer
  contexts help again, up to 3.
- **The squeezed transformer beats its own chain on held-out text**, 5.57
  against 5.88. This is not generalisation. Its approximation error leaks a
  little probability onto tokens the chain forbids, which softens the floor.
  It is the same effect as raising the floor, by accident.

**One split of a 127 KB corpus is a noisy measure.** The configured chain's
held-out NLL by split: random seeds 0 to 5 give 5.88, 5.30, 4.65, 5.31,
5.33, 5.15; the last 10% gives 4.10, the first 10% 5.84, the middle 10%
7.10. Seed 0 is the hardest of the six random splits. The ordering of
methods is the same in every split (the configured chain always loses to
one token of context), so compare methods on the same seed and do not read
much into differences of a few tenths.

Smoothing is a change to the chain only. The compiler takes whatever
distribution each kept context holds, and telescoping does not care how the
rows were made. Two things would need rework: the pruning gain (smoothed
rows differ less from their parents, so thresholds shift) and the output
factoring, which currently leans on each row having few seen tokens.
Smoothed rows are dense.

### Verification (implemented)

- `FidelityMeter`: mean and max KL (reference first), top-1 agreement, largest
  log-probability difference, and both predictors' NLL on the actual next
  token. Reference tokens within 1e-4 nats of the maximum count as tied, so
  rounding an exact tie is not a disagreement.
- The stream is fed in chunks with `max_offset` tokens of lead-in, which
  equals one forward pass (tested).
- Samples from chain and model use the same uniform draws through the inverse
  CDF, so they stay in step exactly as long as the distributions agree.
- With a held-out split, `hegel_compile.py` also prints held-out NLL for
  uniform, unigram, unpruned chain, chain and transformer, and the
  transformer's fidelity to its chain on the held-out stream.

## Open questions

Tracked in `PRODUCT.md`: smoothing the chain, shrinking `d_ff`, structured
token vectors, the closeness threshold, vocabulary size and context length,
summing versus backoff.
