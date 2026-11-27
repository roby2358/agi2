"""Tests for src/verify.py."""

from typing import List

import numpy as np
import pytest
import torch
from src.chain import build_chain, count_contexts
from src.compiler import compile_wide
from src.verify import (
    FidelityMeter,
    compare_chains,
    compare_model_to_chain,
    model_predictor,
    model_stream_log_probs,
    sample,
    unigram_nll,
)


def _stream() -> List[int]:
    rng = np.random.default_rng(7)
    tokens = [0, 1]
    for _ in range(500):
        if rng.random() < 0.75:
            tokens.append((tokens[-1] + tokens[-2] + 1) % 5)
        else:
            tokens.append(int(rng.integers(5)))
    return tokens


@pytest.mark.unit
class TestFidelityMeter:
    def test_identical_distributions(self) -> None:
        logp = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
        meter = FidelityMeter()
        meter.update(logp, logp, np.array([0, -1]))
        result = meter.result()
        assert result.positions == 2
        assert result.kl_mean == 0.0
        assert result.top1_agreement == 1.0
        assert result.reference_nll == pytest.approx(-np.log(0.7))

    def test_known_kl_and_disagreement(self) -> None:
        reference = np.log(np.array([[0.5, 0.5, 1e-12], [0.9, 0.1, 1e-12]]))
        other = np.log(np.array([[0.25, 0.75, 1e-12], [0.1, 0.9, 1e-12]]))
        meter = FidelityMeter()
        meter.update(reference, other, np.array([1, 0]))
        result = meter.result()
        first = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        second = 0.9 * np.log(9.0) + 0.1 * np.log(1 / 9.0)
        assert result.kl_mean == pytest.approx((first + second) / 2)
        assert result.kl_max == pytest.approx(second)
        # Row 0 is a tie in the reference, so either choice agrees.
        assert result.top1_agreement == 0.5
        assert result.other_nll == pytest.approx(-(np.log(0.75) + np.log(0.1)) / 2)


@pytest.mark.unit
class TestComparisons:
    def test_chunked_forward_equals_whole_forward(self) -> None:
        tokens = _stream()
        model = compile_wide(build_chain(count_contexts(tokens, 3), 5, 3))
        whole = model(torch.tensor([tokens]))[0].double().numpy()
        pieces = [lp for _, lp in model_stream_log_probs(model, tokens, chunk=37)]
        assert np.allclose(np.concatenate(pieces), whole, atol=1e-6)

    def test_model_follows_its_chain(self) -> None:
        tokens = _stream()
        chain = build_chain(count_contexts(tokens, 3), 5, 3, prune_threshold=1.0)
        result = compare_model_to_chain(compile_wide(chain), chain, tokens, chunk=64)
        assert result.positions == len(tokens)
        assert result.kl_mean < 1e-5
        assert result.top1_agreement == 1.0

    def test_pruning_costs_likelihood(self) -> None:
        tokens = _stream()
        counts = count_contexts(tokens, 3)
        full = build_chain(counts, 5, 3)
        pruned = build_chain(counts, 5, 3, prune_threshold=4.0)
        assert compare_chains(full, full, tokens).kl_mean == 0.0
        result = compare_chains(full, pruned, tokens)
        assert result.kl_mean > 0.0
        assert result.other_nll > result.reference_nll


@pytest.mark.unit
class TestUnigramNll:
    def test_matches_the_root_row(self) -> None:
        tokens = _stream()
        chain = build_chain(count_contexts(tokens, 2), 5, 2)
        expected = -np.mean([chain.log_probs[0][t] for t in tokens[1:]])
        assert unigram_nll(chain, tokens) == pytest.approx(expected)
        assert unigram_nll(chain, [3]) == 0.0


@pytest.mark.unit
class TestSample:
    def test_inverts_the_cdf(self) -> None:
        logp = np.log(np.array([0.2, 0.5, 0.3]))
        draws = np.array([0.0, 0.19, 0.21, 0.69, 0.71, 0.999])
        assert sample(lambda history: logp, [0], 6, draws) == [0, 0, 1, 1, 2, 2]

    def test_chain_and_model_stay_in_step(self) -> None:
        tokens = _stream()
        chain = build_chain(count_contexts(tokens, 3), 5, 3)
        model = compile_wide(chain)
        draws = np.random.default_rng(0).random(60)
        from_chain = sample(chain.next_log_probs, tokens[:3], 60, draws)
        from_model = sample(model_predictor(model), tokens[:3], 60, draws)
        assert from_chain == from_model
