# Architecture Log

Append-only log of architecture decisions for the semantic (cosine-similarity
training) experiment. Newest entry last. Entries record what changed, why, and
what was measured at the time; the spec and code comments describe only the
current state and point here for history.

---

## 2026-09-03 — Special tokens and stop semantics

Registered `<|endoftext|>`, `<|break|>`, and `<|pad|>` as atomic tokens in
`BPETokenizer` alongside `<EOS>`. Before this, `<|endoftext|>` appeared ~31.6k
times in the dialogue corpus but tokenized as three ordinary subwords, so
models learned to emit the text yet generation never stopped on it. Generation
now stops on a stop-token set (`<EOS>`, `<|endoftext|>`) and strips special
tokens from decoded output. `<|break|>` and `<|pad|>` are reserved, not yet
consumed.

## 2026-09-03 — Boundary-aligned training windows

Training windows snap to `<|endoftext|>` utterance starts (2x-stride
tolerance, stride fill across marker-less stretches) instead of arbitrary
stride offsets, so prompts begin at the start of a speech rather than
mid-sentence. Windows still cross boundaries — dialogue context spans
utterances.

## 2026-09-04 — Chunked WKV with per-chunk gradient checkpointing

The RWKV WKV recurrence is computed in chunks of 16 timesteps with log-space
max-shift stabilization, each chunk wrapped in `torch.utils.checkpoint`
(non-reentrant). Fixed a VRAM blowup that was spilling to system memory under
the WSL2 driver and slowing epochs ~7.5-9x; epochs dropped to ~40s at the
512-window configuration.

## 2026-09-04 — Dense per-position targets

Every real position's hidden state now trains against its next token
(`dense_targets = true`), replacing the original last-position-only
formulation (kept as the `false` control). A window of N tokens yields N
training signals instead of 1 — roughly a 150x increase in signal over a run —
at no extra forward-pass cost, since a causal model computes all the hidden
states anyway.

## 2026-09-04 — InfoNCE objective

Added `objective = "infonce"`: cross-entropy over cosine logits (hidden state
vs every vocab embedding, divided by `nce_temperature` = 0.07). Motivation:
the pairwise sigmoid-gap loss pulled hidden states toward targets but exerted
almost no downward pressure on the other 4,095 vocab entries, so cosine
scores stayed packed and sampling looked near-random even when the
representation ranked the right token highly (measured: `<|endoftext|>` in
the top 3 at true utterance ends with p≈0.018). InfoNCE's softmax
normalization pushes all wrong tokens down and optimizes exactly the
distribution generation samples from. First run produced the project's first
coherent verse (train perplexity 12.42, top1 51.4% — later understood to be a
training-set metric; see the CE control entry). `objective = "pairwise"`
remains the research control.

## 2026-09-05 — Training window widened to 1024, curriculum ramp retired

Window doubled from 512 to 1024 tokens (batch halved to keep the token budget
constant — RWKV has no positional limit, so this is purely a data-distribution
choice; ~16 dialogue exchanges of context instead of ~8). The epoch-by-epoch
seq_len ramp (2 → seq_len_end over the whole run) was then made configurable
(`seq_len_ramp_epochs`) and retired for this model (flat 1024): the ramp was
designed for the one-signal-per-window era, when it was the only source of
short-context lessons. Dense targets moved the curriculum inside each window —
position k of a full-length window IS a k-token-context lesson — so ramping
over the whole run just meant half the epochs contained no long-context
signal at all.

## 2026-09-05 — CE control infrastructure: objective="ce", validation holdout, run logging

Built the missing baseline for the cosine-training thesis plus the apparatus
to compare arms honestly:

- `objective = "ce"`: plain cross-entropy over tied-projection logits on the
  shared dense-target machinery; `scoring = "logits"` generation mode to
  match (a CE model samples from its logit projection at ~temperature 1.0,
  not cosine scores at 0.07).
- `val_fraction`: a boundary-aligned corpus tail held out of training
  entirely, with `val_*` metrics reported every epoch at `seq_len_end`.
- Run persistence: stdout/stderr tee'd to `logs/train-<model>-<timestamp>.log`
  and the history dict dumped to `trained/<model>_history.json` every epoch.

Discovered while building this: **the embedding matrix was never frozen.**
Nothing sets `requires_grad = False` and the optimizer receives all
parameters, so every run to date trained learned embeddings — the spec's
frozen-random-codebook design has never actually been executed. This makes
the CE-vs-InfoNCE comparison objective-only (both arms learn embeddings),
but the original thesis remains untested (task freeze-emb-t3kq).

## 2026-09-05 — Best-validation checkpointing

`trained/<model>_best.pt` is saved whenever the held-out metric improves. The
CE control's val perplexity bottomed at epoch 30 of 100 and tripled by the
end, so the final checkpoint of a long run can be far past the model's best —
and until this change those best weights were lost.

## 2026-09-05 — CE control results: the objective is not the bottleneck

Two arms, identical except the objective (13M-param RWKV, 384x6, BPE-4096,
4.3M-token corpus, flat 1024 windows, dense targets, 100 epochs, same 2%
held-out tail):

| | ce (plain cross-entropy) | infonce (cosine logits) |
|---|---|---|
| best val perplexity | 155.8 (epoch 30) | 165.0 (epoch 32) |
| val perplexity at epoch 100 | 685.9 | 300.8 |
| final train perplexity | 6.1 | 11.0 |

Findings: (1) held-out, the objectives are within ~6% — the cosine-family
objective costs essentially nothing against the standard-LM baseline;
(2) InfoNCE overfits at roughly half the rate — the cosine normalization
acts as a regularizer; (3) both arms hit their val minimum near epoch 30:
the generalization ceiling on this corpus is set by data volume (4.3M
tokens), not by objective, architecture, or window size. It also reframes
the earlier celebrated samples: train perplexity 12.42 coexisted with
mediocre held-out numbers, so their fluency owed much to memorization.

Decisions taken: InfoNCE stays the primary objective; corpus expansion is
the top-priority lever; epoch budget cut to 50 with early stopping switched
to watch the held-out metric (patience 10) — training past the val minimum
is pure memorization. Follow-ups filed: freeze-emb-t3kq (frozen-codebook
third arm), lr-sched-v8mw (warmup + cosine decay).

## 2026-09-05 — Spec normalized to as-built; this log created

SPEC_SIMILARITY_TRAINING.md rewritten to describe the system as it is
(trainable embeddings with the frozen variant marked as planned, dense
per-position targets, boundary-aligned windows, validation holdout) instead
of the original design narrative. Historical rationale, superseded
formulations, and experiment results moved out of spec text and code
comments into this log, which is now the append-only home for both.
