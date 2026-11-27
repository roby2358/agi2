"""Tests for src/compiler.py: the compiled model must reproduce the chain."""

from typing import List

import numpy as np
import pytest
import torch
from src.chain import Chain, build_chain, count_contexts
from src.compiler import compile_chain, compile_wide


def _stream(seed: int, length: int, vocab: int) -> List[int]:
    """A random stream with second-order structure."""
    rng = np.random.default_rng(seed)
    tokens = [0, 1]
    for _ in range(length - 2):
        if rng.random() < 0.7:
            tokens.append((tokens[-1] + 2 * tokens[-2] + 1) % vocab)
        else:
            tokens.append(int(rng.integers(vocab)))
    return tokens


def _max_error(chain: Chain, tokens: List[int], double: bool) -> float:
    model = compile_wide(chain)
    if double:
        # Rebuild in float64 to separate construction error from rounding.
        model = model.double()
        logit = chain.max_context * chain.vocab_size
        log_probs = torch.from_numpy(chain.log_probs)
        model.b_out[logit:] = log_probs[0]
        for neuron in range(model.d_ff):
            row = neuron + 1
            model.w_out[logit:, neuron] = log_probs[row] - log_probs[chain.parents[row]]
    out = model(torch.tensor([tokens]))[0].double().numpy()
    expected = chain.log_probs[chain.match_stream(tokens)]
    return float(np.abs(out - expected).max())


@pytest.mark.unit
class TestCompileWide:
    def test_dimensions(self) -> None:
        tokens = _stream(0, 200, 5)
        chain = build_chain(count_contexts(tokens, 3), 5, 3)
        model = compile_wide(chain)
        assert model.d_model == 4 * 5
        assert model.n_heads == 2
        assert model.d_ff == len(chain.contexts) - 1

    @pytest.mark.parametrize("max_context", [1, 2, 3, 4])
    def test_reproduces_unpruned_chain(self, max_context: int) -> None:
        tokens = _stream(1, 400, 6)
        chain = build_chain(count_contexts(tokens, max_context), 6, max_context)
        assert _max_error(chain, tokens, double=False) < 1e-4
        assert _max_error(chain, tokens, double=True) < 1e-9

    @pytest.mark.parametrize("threshold", [0.5, 2.0, 8.0])
    def test_reproduces_pruned_chain(self, threshold: float) -> None:
        """Pruning leaves gaps in the suffix chain; deltas must still telescope."""
        tokens = _stream(2, 600, 6)
        counts = count_contexts(tokens, 3)
        chain = build_chain(counts, 6, 3, prune_threshold=threshold)
        assert 1 < len(chain.contexts) < len(counts)
        assert _max_error(chain, tokens, double=True) < 1e-9

    def test_unseen_history_backs_off_like_the_chain(self) -> None:
        tokens = _stream(3, 300, 6)
        chain = build_chain(count_contexts(tokens, 3), 6, 3)
        model = compile_wide(chain)
        rng = np.random.default_rng(4)
        fresh = [int(t) for t in rng.integers(6, size=200)]
        out = model(torch.tensor([fresh]))[0].double().numpy()
        expected = chain.log_probs[chain.match_stream(fresh)]
        assert np.abs(out - expected).max() < 1e-4


def _kl(chain: Chain, tokens: List[int], **squeeze: float) -> float:
    model = compile_chain(chain, **squeeze)  # type: ignore[arg-type]
    out = model(torch.tensor([tokens]))[0].double().numpy()
    expected = chain.log_probs[chain.match_stream(tokens)]
    return float(np.mean(np.sum(np.exp(expected) * (expected - out), axis=1)))


@pytest.mark.unit
class TestCompileSqueezed:
    def test_dimensions(self) -> None:
        tokens = _stream(5, 300, 6)
        chain = build_chain(count_contexts(tokens, 3), 6, 3)
        model = compile_chain(chain, slot_dim=10, logit_dim=4)
        assert model.d_model == 3 * 10 + 4
        assert model.d_head == 10
        assert model.d_ff == len(chain.contexts) - 1

    def test_wide_slots_with_threshold_stay_exact(self) -> None:
        tokens = _stream(5, 300, 6)
        chain = build_chain(count_contexts(tokens, 3), 6, 3)
        assert _kl(chain, tokens, match_threshold=0.7) < 1e-6

    def test_roomy_random_slots_reproduce_the_chain(self) -> None:
        tokens = _stream(5, 300, 6)
        chain = build_chain(count_contexts(tokens, 3), 6, 3)
        assert _kl(chain, tokens, slot_dim=64) < 1e-6

    def test_full_rank_logit_code_reproduces_the_chain(self) -> None:
        tokens = _stream(5, 300, 6)
        chain = build_chain(count_contexts(tokens, 3), 6, 3, prune_threshold=1.0)
        assert _kl(chain, tokens, logit_dim=6) < 1e-4

    def test_squeezing_harder_costs_more(self) -> None:
        tokens = _stream(5, 600, 12)
        chain = build_chain(count_contexts(tokens, 3), 12, 3)
        roomy = _kl(chain, tokens, slot_dim=64, logit_dim=10)
        tight = _kl(chain, tokens, slot_dim=64, logit_dim=3)
        assert 0.0 <= roomy < tight
