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

The first milestone is done: the **wide (one-hot) build** runs end to end and
reproduces the chain to float32 rounding. On `../samples/corpus20.txt`
(54,944 tokens, V = 512, context 3, prune threshold 0.5):

- Chain: 24,419 contexts kept of 34,075 (length 1: 330 of 330, length 2:
  8,934 of 9,099, length 3: 15,155 of 24,646).
- Model: `d_model` 2048, 2 heads, `d_ff` 24,419; 106 M weights of which
  668 K are nonzero.
- Transformer against its chain, all positions: mean KL 3.0e-7 nats, top-1
  agreement 100%, largest log-probability difference 3.2e-6. Samples drawn
  with shared random numbers are token-for-token identical.
- The whole run takes about 40 s on CPU, nearly all of it the verification
  pass through the dense feed-forward layer.

Next step: the **squeezed build**. Scratch experiments (not in the repo) give
the starting point; see "Squeezed build" below.

## Files and commands

```
hegel_compile.py        entry point: tokenize, chain, compile, verify, sample
resources/corpus20.toml the first-milestone config (all keys: src/config.py)
src/config.py           HegelConfig, load_config (TOML, unknown keys rejected)
src/corpus.py           normalize (one stream), tokenize (BPE from semantic/)
src/chain.py            count_contexts, build_chain (pruning), Chain (backoff)
src/model.py            HegelModel: the minimal transformer, all weights zero
src/compiler.py         compile_wide: Chain -> HegelModel
src/verify.py           FidelityMeter, chain/model comparisons, coupled sampling
tests/                  one test file per src module
```

```bash
just hegel                 # = just hegel corpus20, from anywhere in the repo
cd hegel
uv run python hegel_compile.py resources/corpus20.toml
uv run pytest tests        # 46 tests, about 15 s
uv run black src tests hegel_compile.py
uv run isort src tests hegel_compile.py
uv run flake8 --max-line-length 88 --extend-ignore E203 src tests hegel_compile.py
uv run mypy src hegel_compile.py
```

Run from `hegel/`: scripts and tests import `src.*` relative to it.

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

### Squeezed build (the real target; first scratch numbers only)

Measured with the 0.5-pruned chain, one side squeezed at a time.

- **Input side squeezes almost for free.** Replace each one-hot slot with a
  random unit vector of d dims. A context neuron takes the dot product with
  its tokens' vectors and fires through a raised threshold:
  `relu((score - (length - 1) - tau) / (1 - tau))`. With outputs kept exact:
  d = 32, tau = 0.7 gives mean KL below 1e-4 and 100% top-1; d = 16,
  tau = 0.7 gives KL 0.02; d = 64, tau = 0.5 is exact to 4 decimals. Too low
  a threshold is catastrophic (d = 32, tau = 0.3: KL 4.7), because near-miss
  contexts fire by the thousand.
- **Output side does not squeeze by plain SVD.** Projecting the logits onto
  the top-r right singular vectors of the log-probability table: r = 256
  gives KL 0.28 to 0.33, r = 128 gives 1.9, r = 64 gives 3.4; r = 384 is
  exact (only about 330 distinct tokens ever follow anything). Centring
  against the unigram row, the floor, or the row mean makes no difference, and
  neither does weighting rows by frequency. The distributions are nearly
  one-hot, which is the worst case for a least-squares fit in log space.
- **A random decoder beats SVD at low rank** (r = 64: KL 0.76, top-1 83%;
  r = 128: KL 0.49) but plateaus near 87% top-1.
- So a half squeeze is available now: slots of 32 dims with a full V-dim
  logit region gives `d_model` about 3 * 32 + 512 = 608 instead of 2048, at
  no measurable cost. The open problem is the output rank.
- Near-miss firing with structured (not random) token vectors is where
  generalisation would come from. Not tried yet.

### Verification (implemented)

- `FidelityMeter`: mean and max KL (reference first), top-1 agreement, largest
  log-probability difference, and both predictors' NLL on the actual next
  token. Reference tokens within 1e-4 nats of the maximum count as tied, so
  rounding an exact tie is not a disagreement.
- The stream is fed in chunks with `max_offset` tokens of lead-in, which
  equals one forward pass (tested).
- Samples from chain and model use the same uniform draws through the inverse
  CDF, so they stay in step exactly as long as the distributions agree.

## Open questions

Tracked in `PRODUCT.md`: how to squeeze the output side, the closeness
threshold for a lossy build, vocabulary size and context length, summing
versus backoff.
