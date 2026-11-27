"""Tests for src/factor.py."""

import numpy as np
import pytest
from src.chain import build_chain, count_contexts
from src.factor import factor_table, fit_decoder, fit_logit_codes, random_codes


def _table() -> tuple[np.ndarray, float]:  # type: ignore[type-arg]
    rng = np.random.default_rng(5)
    tokens = [int(t) for t in rng.integers(8, size=300)]
    tokens += [0, 1, 2, 3] * 40
    chain = build_chain(count_contexts(tokens, 2), 8, 2, floor=1e-3)
    return chain.log_probs, float(np.log(1e-3 / 8))


def _softmax_log(logits: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    shifted = logits - logits.max(axis=1, keepdims=True)
    result: np.ndarray = shifted - np.log(  # type: ignore[type-arg]
        np.exp(shifted).sum(axis=1, keepdims=True)
    )
    return result


@pytest.mark.unit
class TestRandomCodes:
    def test_unit_length_and_reproducible(self) -> None:
        codes = random_codes(20, 7, seed=3)
        assert codes.shape == (20, 7)
        assert np.allclose(np.linalg.norm(codes, axis=1), 1.0)
        assert np.array_equal(codes, random_codes(20, 7, seed=3))
        assert not np.array_equal(codes, random_codes(20, 7, seed=4))


@pytest.mark.unit
class TestFitLogitCodes:
    @pytest.mark.parametrize("unseen_weight", [1.0, 0.01])
    def test_full_rank_is_exact(self, unseen_weight: float) -> None:
        log_probs, floor = _table()
        decoder = random_codes(8, 8, seed=0)
        codes = fit_logit_codes(log_probs, floor, decoder, unseen_weight)
        assert np.allclose(_softmax_log(codes @ decoder.T), log_probs, atol=1e-5)

    def test_chunking_does_not_change_the_result(self) -> None:
        log_probs, floor = _table()
        decoder = random_codes(8, 4, seed=1)
        whole = fit_logit_codes(log_probs, floor, decoder, 0.05)
        pieces = fit_logit_codes(
            log_probs, floor, decoder, 0.05, chunk_bytes=8 * 16 * 3
        )
        assert np.allclose(whole, pieces)


def _weighted_error(
    log_probs: np.ndarray,  # type: ignore[type-arg]
    floor: float,
    decoder: np.ndarray,  # type: ignore[type-arg]
    codes: np.ndarray,  # type: ignore[type-arg]
    unseen_weight: float,
) -> float:
    seen = log_probs > floor + 1e-9
    target = seen * (log_probs - floor)
    weight = np.where(seen, 1.0, unseen_weight)
    return float(np.sum(weight * (codes @ decoder.T - target) ** 2))


@pytest.mark.unit
class TestFactorTable:
    def test_zero_rounds_is_the_random_decoder(self) -> None:
        log_probs, floor = _table()
        weights = np.ones(len(log_probs))
        decoder, codes = factor_table(log_probs, floor, 4, 0.01, weights, 0, seed=2)
        assert np.array_equal(decoder, random_codes(8, 4, seed=2))
        assert np.allclose(codes, fit_logit_codes(log_probs, floor, decoder, 0.01))

    def test_each_round_lowers_the_weighted_error(self) -> None:
        log_probs, floor = _table()
        weights = np.ones(len(log_probs))
        errors = []
        for rounds in range(4):
            decoder, codes = factor_table(
                log_probs, floor, 4, 0.05, weights, rounds, seed=2
            )
            errors.append(_weighted_error(log_probs, floor, decoder, codes, 0.05))
        assert errors[0] > errors[1] > errors[2] > errors[3]

    def test_decoder_step_is_the_least_squares_optimum(self) -> None:
        log_probs, floor = _table()
        rng = np.random.default_rng(0)
        weights = rng.random(len(log_probs)) + 0.5
        codes = rng.standard_normal((len(log_probs), 3))
        decoder = fit_decoder(log_probs, floor, codes, 0.1, weights)
        seen = log_probs > floor + 1e-9
        target = seen * (log_probs - floor)
        for token in range(8):
            scale = np.sqrt(weights * np.where(seen[:, token], 1.0, 0.1))
            expected = np.linalg.lstsq(
                codes * scale[:, None], target[:, token] * scale, rcond=None
            )[0]
            assert np.allclose(decoder[token], expected, atol=1e-6)
