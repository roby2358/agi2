"""
Low-rank output factoring: next-token distributions through a narrow decoder.

A plain SVD of the log-probability table fails here, because the rows are
nearly one-hot and an even-handed least-squares fit in log space spends its
rank on the thousands of floor entries. This module solves a weighted
problem instead: every context wants its seen tokens' logits right and its
unseen tokens merely low.

    minimise  sum over contexts c, tokens v of
              row_weight[c] * token_weight[c, v]
              * (code[c] . decoder[v] - target[c, v])^2

with target = log P(v | c) - floor on seen tokens and 0 on unseen ones, and
token_weight = 1 on seen tokens, unseen_weight on unseen ones.

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


def random_codes(count: int, dim: int, seed: int) -> Matrix:
    """One random unit vector of `dim` dimensions per item."""
    rng = np.random.default_rng(seed)
    codes = rng.standard_normal((count, dim))
    unit: Matrix = codes / np.linalg.norm(codes, axis=1, keepdims=True)
    return unit


def _seen_and_target(
    log_probs: Matrix, floor_log_prob: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    table = torch.from_numpy(log_probs)
    seen = table > floor_log_prob + 1e-9
    return seen, seen.double() * (table - floor_log_prob)


def _solve_side(
    seen: torch.Tensor,
    rhs: torch.Tensor,
    other: torch.Tensor,
    other_weights: torch.Tensor,
    unseen_weight: float,
    chunk_bytes: int,
) -> torch.Tensor:
    """
    Solve one side of the factoring, one row of `seen` at a time.

    Row i's normal matrix is
        unseen_weight * sum_j w_j o_j o_j^T
        + (1 - unseen_weight) * sum over seen j of w_j o_j o_j^T
    where o_j are the rows of `other` and w_j their weights. The second sum
    runs only over the few seen entries, which is what keeps this fast.
    """
    count, rank = seen.shape[0], other.shape[1]
    weighted = other * other_weights[:, None]
    base = unseen_weight * (weighted.T @ other)
    base = base + 1e-9 * torch.eye(rank, dtype=torch.float64)
    solved = torch.empty(count, rank, dtype=torch.float64)
    chunk = max(1, chunk_bytes // (8 * rank * rank))
    for start in range(0, count, chunk):
        block = seen[start : start + chunk]
        normal = torch.zeros(block.shape[0], rank * rank, dtype=torch.float64)
        hits = block.nonzero()
        for piece in range(0, len(hits), chunk):
            part = hits[piece : piece + chunk]
            left, right = weighted[part[:, 1]], other[part[:, 1]]
            outer = (left[:, :, None] * right[:, None, :]).reshape(-1, rank * rank)
            normal.index_add_(0, part[:, 0], outer)
        normal = base[None] + (1.0 - unseen_weight) * normal.reshape(-1, rank, rank)
        solved[start : start + chunk] = torch.linalg.solve(
            normal, rhs[start : start + chunk]
        )
    return solved


def fit_logit_codes(
    log_probs: Matrix,
    floor_log_prob: float,
    decoder: Matrix,
    unseen_weight: float,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> Matrix:
    """
    For every context, the code whose decoded logits best match its row.

    Args:
        log_probs: Next-token log-probabilities, shape (contexts, vocab).
        floor_log_prob: The log-probability of a never-seen token.
        decoder: One vector per token, shape (vocab, rank).
        unseen_weight: Weight of an unseen token's error; 1 is plain least
            squares.

    Returns:
        Codes, shape (contexts, rank). `codes @ decoder.T` are the logits.
    """
    seen, target = _seen_and_target(log_probs, floor_log_prob)
    basis = torch.from_numpy(decoder)
    ones = torch.ones(basis.shape[0], dtype=torch.float64)
    codes = _solve_side(seen, target @ basis, basis, ones, unseen_weight, chunk_bytes)
    result: Matrix = codes.numpy()
    return result


def fit_decoder(
    log_probs: Matrix,
    floor_log_prob: float,
    codes: Matrix,
    unseen_weight: float,
    row_weights: Matrix,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> Matrix:
    """
    For every token, the decoder vector that best serves the given codes.

    Args:
        log_probs: Next-token log-probabilities, shape (contexts, vocab).
        floor_log_prob: The log-probability of a never-seen token.
        codes: One code per context, shape (contexts, rank).
        unseen_weight: Weight of an unseen token's error.
        row_weights: How much each context's error counts, shape (contexts,).

    Returns:
        Decoder, shape (vocab, rank).
    """
    seen, target = _seen_and_target(log_probs, floor_log_prob)
    rows = torch.from_numpy(codes)
    weights = torch.from_numpy(row_weights)
    rhs = (target * weights[:, None]).T @ rows
    decoder = _solve_side(
        seen.T.contiguous(), rhs, rows, weights, unseen_weight, chunk_bytes
    )
    result: Matrix = decoder.numpy()
    return result


def factor_table(
    log_probs: Matrix,
    floor_log_prob: float,
    rank: int,
    unseen_weight: float,
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
    codes = fit_logit_codes(log_probs, floor_log_prob, decoder, unseen_weight)
    for _ in range(rounds):
        decoder = fit_decoder(
            log_probs, floor_log_prob, codes, unseen_weight, row_weights
        )
        codes = fit_logit_codes(log_probs, floor_log_prob, decoder, unseen_weight)
    return decoder, codes
