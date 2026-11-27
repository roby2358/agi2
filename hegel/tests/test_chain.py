"""Tests for src/chain.py."""

import numpy as np
import pytest
from src.chain import build_chain, count_contexts

# 0 1 2 repeated, then one 0 1 3: after "0 1" the next token is usually 2.
STREAM = [0, 1, 2] * 6 + [0, 1, 3]


@pytest.mark.unit
class TestCountContexts:
    def test_counts_by_length(self) -> None:
        counts = count_contexts([0, 1, 0, 1, 2], 2)
        assert counts[()] == {1: 2, 0: 1, 2: 1}
        assert counts[(0,)] == {1: 2}
        assert counts[(1,)] == {0: 1, 2: 1}
        assert counts[(0, 1)] == {0: 1, 2: 1}
        assert counts[(1, 0)] == {1: 1}

    def test_last_token_has_no_successor(self) -> None:
        assert (5,) not in count_contexts([4, 5], 2)

    def test_respects_max_context(self) -> None:
        counts = count_contexts(STREAM, 2)
        assert max(len(context) for context in counts) == 2


@pytest.mark.unit
class TestBuildChain:
    def test_rows_are_distributions(self) -> None:
        chain = build_chain(count_contexts(STREAM, 3), 4, 3, floor=1e-3)
        assert np.allclose(np.exp(chain.log_probs).sum(axis=1), 1.0)
        assert np.all(np.isfinite(chain.log_probs))

    def test_floor_and_frequencies(self) -> None:
        chain = build_chain(count_contexts(STREAM, 3), 4, 3, floor=1e-3)
        probs = np.exp(chain.next_log_probs([0, 1]))
        assert probs[0] == pytest.approx(1e-3 / 4)
        assert probs[2] == pytest.approx(0.999 * 6 / 7 + 1e-3 / 4)
        assert probs[3] == pytest.approx(0.999 * 1 / 7 + 1e-3 / 4)

    def test_longest_match_wins(self) -> None:
        chain = build_chain(count_contexts(STREAM, 3), 4, 3)
        assert chain.contexts[chain.match([1, 2, 0, 1])] == (2, 0, 1)
        assert chain.contexts[chain.match([0, 1])] == (0, 1)

    def test_backs_off_to_shorter_then_root(self) -> None:
        chain = build_chain(count_contexts(STREAM, 3), 4, 3)
        # "3 0 1" never occurs; "0 1" does.
        assert chain.contexts[chain.match([3, 0, 1])] == (0, 1)
        # 3 is the last token, so it was never a context.
        assert chain.match([3]) == 0
        assert chain.match([]) == 0

    def test_match_stream_agrees_with_match(self) -> None:
        chain = build_chain(count_contexts(STREAM, 3), 4, 3)
        matches = chain.match_stream(STREAM)
        for i in range(len(STREAM)):
            assert matches[i] == chain.match(STREAM[: i + 1])

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 2.0])
    def test_match_counts_agree_with_the_stream(self, threshold: float) -> None:
        rng = np.random.default_rng(3)
        tokens = [int(t) for t in rng.integers(4, size=400)] + [0, 1, 2] * 50
        chain = build_chain(count_contexts(tokens, 3), 4, 3, prune_threshold=threshold)
        # The last position predicts nothing, so it was never counted.
        expected = np.bincount(
            chain.match_stream(tokens)[:-1], minlength=len(chain.contexts)
        )
        assert np.array_equal(chain.match_counts(), expected)

    def test_unpruned_keeps_everything(self) -> None:
        counts = count_contexts(STREAM, 3)
        chain = build_chain(counts, 4, 3)
        assert len(chain.contexts) == len(counts)
        assert chain.kept_by_length() == chain.candidates

    def test_pruning_drops_contexts_that_add_nothing(self) -> None:
        # A deterministic cycle: one token of context already says it all.
        counts = count_contexts([0, 1, 2] * 10, 3)
        chain = build_chain(counts, 3, 3, prune_threshold=0.5)
        assert chain.kept_by_length() == [1, 3, 0, 0]

    def test_pruning_keeps_contexts_that_matter(self) -> None:
        # Second-order: the token after 0 depends on what preceded the 0.
        counts = count_contexts([1, 0, 1, 2, 0, 2] * 10, 2)
        chain = build_chain(counts, 3, 2, prune_threshold=0.5)
        assert (1, 0) in chain.contexts and (2, 0) in chain.contexts

    def test_unknown_smoothing_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_chain(count_contexts(STREAM, 2), 4, 2, smoothing="laplace")

    def test_witten_bell_rows_are_distributions(self) -> None:
        chain = build_chain(count_contexts(STREAM, 3), 4, 3, smoothing="witten_bell")
        assert chain.smoothing == "witten_bell"
        assert np.allclose(np.exp(chain.log_probs).sum(axis=1), 1.0)

    def test_witten_bell_mixes_with_the_shorter_context(self) -> None:
        counts = count_contexts(STREAM, 2)
        chain = build_chain(counts, 4, 2, floor=1e-3, smoothing="witten_bell")
        root = np.exp(chain.next_log_probs([]))
        # After 1 comes 2 six times and 3 once: 7 occurrences, 2 distinct.
        own = 7 / 9
        one = np.exp(chain.next_log_probs([1]))
        assert one[2] == pytest.approx(own * 6 / 7 + (1 - own) * root[2])
        assert one[0] == pytest.approx((1 - own) * root[0])
        # "0 1" has the same counts and mixes with the row for "1".
        two = np.exp(chain.next_log_probs([0, 1]))
        assert two[3] == pytest.approx(own * 1 / 7 + (1 - own) * one[3])

    def test_witten_bell_hedges_a_context_seen_once(self) -> None:
        tokens = [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 1, 0]
        hard = build_chain(count_contexts(tokens, 2), 4, 2)
        soft = build_chain(count_contexts(tokens, 2), 4, 2, smoothing="witten_bell")
        # "3 1" was seen once, followed by 0. How likely is 2 instead?
        assert np.exp(hard.next_log_probs([3, 1]))[2] < 1e-4
        assert np.exp(soft.next_log_probs([3, 1]))[2] > 0.2

    def test_witten_bell_uses_pruned_contexts_in_the_mix(self) -> None:
        rng = np.random.default_rng(1)
        tokens = [int(t) for t in rng.integers(4, size=300)]
        counts = count_contexts(tokens, 3)
        full = build_chain(counts, 4, 3, smoothing="witten_bell")
        pruned = build_chain(counts, 4, 3, prune_threshold=3.0, smoothing="witten_bell")
        assert len(pruned.contexts) < len(full.contexts)
        for i, context in enumerate(pruned.contexts):
            j = full.contexts.index(context)
            assert np.allclose(pruned.log_probs[i], full.log_probs[j])

    def test_parent_is_longest_kept_proper_suffix(self) -> None:
        chain = build_chain(count_contexts(STREAM, 3), 4, 3, prune_threshold=0.5)
        kept = set(chain.contexts)
        for context, parent in zip(chain.contexts[1:], chain.parents[1:]):
            suffix = chain.contexts[parent]
            assert len(suffix) < len(context)
            assert context[len(context) - len(suffix) :] == suffix
            for length in range(len(suffix) + 1, len(context)):
                assert context[len(context) - length :] not in kept
        assert chain.parents[0] == -1
