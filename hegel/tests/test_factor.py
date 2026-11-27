"""Tests for src/factor.py."""

import numpy as np
import numpy.typing as npt
import pytest
from src.chain import build_chain, count_contexts
from src.factor import (
    factor_table,
    fit_decoder,
    fit_logit_codes,
    random_codes,
    targets_and_weights,
)

Matrix = npt.NDArray[np.float64]


def _table(smoothing: str = "none") -> Matrix:
    rng = np.random.default_rng(5)
    tokens = [int(t) for t in rng.integers(8, size=300)]
    tokens += [0, 1, 2, 3] * 40
    chain = build_chain(count_contexts(tokens, 2), 8, 2, 1e-3, smoothing=smoothing)
    return chain.log_probs


def _softmax_log(logits: Matrix) -> Matrix:
    shifted = logits - logits.max(axis=1, keepdims=True)
    result: Matrix = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return result


def _weighted_error(
    log_probs: Matrix, decoder: Matrix, codes: Matrix, low_weight: float
) -> float:
    target, weight = targets_and_weights(log_probs, low_weight)
    return float(np.sum(weight * (codes @ decoder.T - target) ** 2))


@pytest.mark.unit
class TestRandomCodes:
    def test_unit_length_and_reproducible(self) -> None:
        codes = random_codes(20, 7, seed=3)
        assert codes.shape == (20, 7)
        assert np.allclose(np.linalg.norm(codes, axis=1), 1.0)
        assert np.array_equal(codes, random_codes(20, 7, seed=3))
        assert not np.array_equal(codes, random_codes(20, 7, seed=4))


@pytest.mark.unit
class TestTargetsAndWeights:
    def test_lowest_token_is_zero_and_top_token_weighs_most(self) -> None:
        log_probs = np.log(np.array([[0.7, 0.2, 0.1], [0.05, 0.9, 0.05]]))
        target, weight = targets_and_weights(log_probs, 0.01)
        assert np.allclose(target.min(axis=1), 0.0)
        assert np.allclose(target[0], np.log([7.0, 2.0, 1.0]))
        assert np.allclose(weight[0], [1.01, 2 / 7 + 0.01, 1 / 7 + 0.01])
        assert np.allclose(weight.max(axis=1), 1.01)


@pytest.mark.unit
class TestFitLogitCodes:
    @pytest.mark.parametrize("smoothing", ["none", "witten_bell"])
    @pytest.mark.parametrize("low_weight", [1.0, 0.003])
    def test_full_rank_is_exact(self, smoothing: str, low_weight: float) -> None:
        log_probs = _table(smoothing)
        decoder = random_codes(8, 8, seed=0)
        codes = fit_logit_codes(log_probs, decoder, low_weight)
        assert np.allclose(_softmax_log(codes @ decoder.T), log_probs, atol=1e-4)

    def test_is_the_weighted_least_squares_optimum(self) -> None:
        log_probs = _table()
        decoder = random_codes(8, 3, seed=1)
        codes = fit_logit_codes(log_probs, decoder, 0.05)
        target, weight = targets_and_weights(log_probs, 0.05)
        for row in range(0, len(log_probs), 7):
            scale = np.sqrt(weight[row])
            expected = np.linalg.lstsq(
                decoder * scale[:, None], target[row] * scale, rcond=None
            )[0]
            assert np.allclose(codes[row], expected, atol=1e-6)


@pytest.mark.unit
class TestFitDecoder:
    def test_is_the_weighted_least_squares_optimum(self) -> None:
        log_probs = _table()
        rng = np.random.default_rng(0)
        row_weights = rng.random(len(log_probs)) + 0.5
        codes = rng.standard_normal((len(log_probs), 3))
        decoder = fit_decoder(log_probs, codes, 0.1, row_weights)
        target, weight = targets_and_weights(log_probs, 0.1)
        for token in range(8):
            scale = np.sqrt(row_weights * weight[:, token])
            expected = np.linalg.lstsq(
                codes * scale[:, None], target[:, token] * scale, rcond=None
            )[0]
            assert np.allclose(decoder[token], expected, atol=1e-6)


@pytest.mark.unit
class TestFactorTable:
    def test_zero_rounds_is_the_random_decoder(self) -> None:
        log_probs = _table()
        weights = np.ones(len(log_probs))
        decoder, codes = factor_table(log_probs, 4, 0.01, weights, 0, seed=2)
        assert np.array_equal(decoder, random_codes(8, 4, seed=2))
        assert np.allclose(codes, fit_logit_codes(log_probs, decoder, 0.01))

    @pytest.mark.parametrize("smoothing", ["none", "witten_bell"])
    def test_each_round_lowers_the_weighted_error(self, smoothing: str) -> None:
        log_probs = _table(smoothing)
        weights = np.ones(len(log_probs))
        errors = []
        for rounds in range(4):
            decoder, codes = factor_table(log_probs, 4, 0.05, weights, rounds, seed=2)
            errors.append(_weighted_error(log_probs, decoder, codes, 0.05))
        assert errors[0] > errors[1] > errors[2] > errors[3]
