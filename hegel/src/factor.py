"""
Low-rank output factoring: next-token distributions through a narrow decoder.

A plain SVD of the log-probability table fails here: an even-handed
least-squares fit in log space spends its rank on the many tokens a context
gives almost no probability, and gets the few that matter wrong. This module
solves a weighted problem instead:

    minimise  sum over contexts c, tokens v of
              row_weight[c] * token_weight[c, v]
              * (code[c] . decoder[v] - target[c, v])^2

- target[c, v] = log P(v | c) minus the row's lowest log-probability, so the
  least likely tokens sit at 0 (softmax ignores the shared offset) and
  0, the answer an untouched direction gives, means "unlikely".
- token_weight[c, v] = P(v | c) / max_v P(v | c) + low_weight. The tokens a
  context actually predicts are fitted closely; the rest only need to stay
  low on the whole.

Given the decoder, each context's code is one closed-form solve; given the
codes, each token's decoder vector is one closed-form solve. Alternating the
two (alternating least squares) is a factoring of the table. It never runs
the model, and takes no gradient step.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
import torch

Matrix = npt.NDArray[np.float64]

CHUNK_BYTES = 64 * 1024 * 1024


def random_codes(count: int, dim: int, seed: int) -> Matrix:
    """One random unit vector of `dim` dimensions per item."""
    rng = np.random.default_rng(seed)
    codes = rng.standard_normal((count, dim))
    unit: Matrix = codes / np.linalg.norm(codes, axis=1, keepdims=True)
    return unit


def targets_and_weights(log_probs: Matrix, low_weight: float) -> Tuple[Matrix, Matrix]:
    """The fit's target logits and per-token weights; see the module docstring."""
    target: Matrix = log_probs - log_probs.min(axis=1, keepdims=True)
    weight: Matrix = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
    return target, weight + low_weight


def _solve_side(
    weight: torch.Tensor, rhs: torch.Tensor, other: torch.Tensor
) -> torch.Tensor:
    """
    Solve one side of the factoring.

    Row i's normal matrix is sum_j weight[i, j] * o_j o_j^T, where o_j are
    the rows of `other`; its right-hand side is rhs[i].
    """
    rank = other.shape[1]
    outer = (other[:, :, None] * other[:, None, :]).reshape(-1, rank * rank)
    ridge = 1e-9 * torch.eye(rank, dtype=torch.float64)
    solved = torch.empty(weight.shape[0], rank, dtype=torch.float64)
    chunk = max(1, CHUNK_BYTES // (8 * rank * rank))
    for start in range(0, weight.shape[0], chunk):
        normal = (weight[start : start + chunk] @ outer).reshape(-1, rank, rank)
        solved[start : start + chunk] = torch.linalg.solve(
            normal + ridge, rhs[start : start + chunk]
        )
    return solved


def fit_logit_codes(log_probs: Matrix, decoder: Matrix, low_weight: float) -> Matrix:
    """
    For every context, the code whose decoded logits best match its row.

    Args:
        log_probs: Next-token log-probabilities, shape (contexts, vocab).
        decoder: One vector per token, shape (vocab, rank).
        low_weight: Weight added to every token; what a token with no
            probability still counts for.

    Returns:
        Codes, shape (contexts, rank). `codes @ decoder.T` are the logits.
    """
    target, weight = (
        torch.from_numpy(m) for m in targets_and_weights(log_probs, low_weight)
    )
    basis = torch.from_numpy(decoder)
    codes = _solve_side(weight, (weight * target) @ basis, basis)
    result: Matrix = codes.numpy()
    return result


def fit_decoder(
    log_probs: Matrix, codes: Matrix, low_weight: float, row_weights: Matrix
) -> Matrix:
    """
    For every token, the decoder vector that best serves the given codes.

    Args:
        log_probs: Next-token log-probabilities, shape (contexts, vocab).
        codes: One code per context, shape (contexts, rank).
        low_weight: Weight added to every token.
        row_weights: How much each context's error counts, shape (contexts,).

    Returns:
        Decoder, shape (vocab, rank).
    """
    target, weight = (
        torch.from_numpy(m) for m in targets_and_weights(log_probs, low_weight)
    )
    rows = torch.from_numpy(codes)
    weight = weight * torch.from_numpy(row_weights)[:, None]
    rank = rows.shape[1]
    normal = torch.zeros(weight.shape[1], rank * rank, dtype=torch.float64)
    chunk = max(1, CHUNK_BYTES // (8 * rank * rank))
    for start in range(0, rows.shape[0], chunk):
        part = rows[start : start + chunk]
        outer = (part[:, :, None] * part[:, None, :]).reshape(-1, rank * rank)
        normal += weight[start : start + chunk].T @ outer
    ridge = 1e-9 * torch.eye(rank, dtype=torch.float64)
    decoder = torch.linalg.solve(
        normal.reshape(-1, rank, rank) + ridge, (weight * target).T @ rows
    )
    result: Matrix = decoder.numpy()
    return result


def factor_table(
    log_probs: Matrix,
    rank: int,
    low_weight: float,
    row_weights: Matrix,
    rounds: int,
    seed: int,
) -> Tuple[Matrix, Matrix]:
    """
    Factor the table as codes @ decoder.T at the given rank.

    Starts from a random unit-vector decoder and fits the codes. Each round
    then refits the decoder to the codes and the codes to the decoder.
    rounds = 0 keeps the random decoder.

    Returns:
        (decoder of shape (vocab, rank), codes of shape (contexts, rank)).
    """
    decoder = random_codes(log_probs.shape[1], rank, seed)
    codes = fit_logit_codes(log_probs, decoder, low_weight)
    for _ in range(rounds):
        decoder = fit_decoder(log_probs, codes, low_weight, row_weights)
        codes = fit_logit_codes(log_probs, decoder, low_weight)
    return decoder, codes
