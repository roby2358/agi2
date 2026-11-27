"""Compare next-token distributions, and sample chain and model side by side."""

from dataclasses import dataclass
from typing import Callable, Iterator, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch
from src.chain import Chain
from src.model import HegelModel

LogProbs = npt.NDArray[np.float64]

# Reference tokens within this many nats of the reference maximum count as
# tied for top-1, so float rounding of an exact tie is not a disagreement.
TIE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Fidelity:
    """
    How closely one predictor follows a reference, averaged over positions.

    Attributes:
        positions: Number of positions compared.
        kl_mean: Mean KL(reference || other) in nats.
        kl_max: Largest KL at any single position.
        top1_agreement: Fraction of positions where the other's most likely
            token is (tied for) the reference's most likely token.
        max_abs_diff: Largest absolute difference of any log-probability.
        reference_nll: Mean negative log-likelihood of the actual next token
            under the reference, in nats.
        other_nll: The same under the other predictor.
        reference_accuracy: Fraction of positions where the reference's most
            likely token is the actual next token.
        other_accuracy: The same for the other predictor.
    """

    positions: int
    kl_mean: float
    kl_max: float
    top1_agreement: float
    max_abs_diff: float
    reference_nll: float
    other_nll: float
    reference_accuracy: float
    other_accuracy: float


class FidelityMeter:
    """Accumulates a Fidelity over batches of positions."""

    def __init__(self) -> None:
        self._positions = 0
        self._kl_sum = 0.0
        self._kl_max = 0.0
        self._agree = 0
        self._max_abs_diff = 0.0
        self._scored = 0
        self._reference_nll = 0.0
        self._other_nll = 0.0
        self._reference_hits = 0
        self._other_hits = 0

    def update(
        self,
        reference: LogProbs,
        other: LogProbs,
        targets: npt.NDArray[np.int64],
    ) -> None:
        """
        Add a batch.

        Args:
            reference: Log-probabilities, shape (positions, vocab).
            other: Log-probabilities to compare, same shape.
            targets: The actual next token per position, or -1 where there
                is none.
        """
        kl = np.sum(np.exp(reference) * (reference - other), axis=1)
        self._positions += len(kl)
        self._kl_sum += float(kl.sum())
        self._kl_max = max(self._kl_max, float(kl.max()))
        rows = np.arange(len(kl))
        picked = reference[rows, other.argmax(axis=1)]
        self._agree += int(np.sum(picked >= reference.max(axis=1) - TIE_TOLERANCE))
        self._max_abs_diff = max(
            self._max_abs_diff, float(np.abs(reference - other).max())
        )
        scored = targets >= 0
        self._scored += int(scored.sum())
        self._reference_nll -= float(reference[rows[scored], targets[scored]].sum())
        self._other_nll -= float(other[rows[scored], targets[scored]].sum())
        wanted = targets[scored]
        self._reference_hits += int(np.sum(reference[scored].argmax(axis=1) == wanted))
        self._other_hits += int(np.sum(other[scored].argmax(axis=1) == wanted))

    def result(self) -> Fidelity:
        """The accumulated Fidelity."""
        positions = max(self._positions, 1)
        scored = max(self._scored, 1)
        return Fidelity(
            positions=self._positions,
            kl_mean=self._kl_sum / positions,
            kl_max=self._kl_max,
            top1_agreement=self._agree / positions,
            max_abs_diff=self._max_abs_diff,
            reference_nll=self._reference_nll / scored,
            other_nll=self._other_nll / scored,
            reference_accuracy=self._reference_hits / scored,
            other_accuracy=self._other_hits / scored,
        )


def _targets(tokens: Sequence[int]) -> npt.NDArray[np.int64]:
    """The next token at each position; -1 at the last."""
    targets = np.full(len(tokens), -1, dtype=np.int64)
    targets[:-1] = np.asarray(tokens[1:], dtype=np.int64)
    return targets


def model_stream_log_probs(
    model: HegelModel, tokens: Sequence[int], chunk: int = 1024
) -> Iterator[Tuple[int, LogProbs]]:
    """
    Run the model over a long stream in chunks.

    Each chunk is fed with the max_offset tokens before it, which is all the
    model can see, so the result equals one forward pass over the stream.

    Yields:
        (start position, log-probabilities for positions start..start+chunk).
    """
    ids = torch.as_tensor(list(tokens), dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(tokens), chunk):
            lead = min(start, model.max_offset)
            window = ids[start - lead : start + chunk]
            out = model(window[None, :])[0, lead:]
            yield start, out.double().numpy()


def compare_model_to_chain(
    model: HegelModel, chain: Chain, tokens: Sequence[int], chunk: int = 1024
) -> Fidelity:
    """Fidelity of the model to the chain at every position of the stream."""
    matches = chain.match_stream(tokens)
    targets = _targets(tokens)
    meter = FidelityMeter()
    for start, model_lp in model_stream_log_probs(model, tokens, chunk):
        stop = start + len(model_lp)
        meter.update(
            chain.log_probs[matches[start:stop]], model_lp, targets[start:stop]
        )
    return meter.result()


def compare_chains(
    reference: Chain, other: Chain, tokens: Sequence[int], chunk: int = 4096
) -> Fidelity:
    """Fidelity of one chain to another at every position of the stream."""
    reference_matches = reference.match_stream(tokens)
    other_matches = other.match_stream(tokens)
    targets = _targets(tokens)
    meter = FidelityMeter()
    for start in range(0, len(tokens), chunk):
        stop = start + chunk
        meter.update(
            reference.log_probs[reference_matches[start:stop]],
            other.log_probs[other_matches[start:stop]],
            targets[start:stop],
        )
    return meter.result()


def chain_score(chain: Chain, tokens: Sequence[int]) -> Tuple[float, float]:
    """
    How well a chain predicts a stream.

    Returns:
        (mean NLL in nats of the actual next token, fraction of positions
        where the chain's most likely token is the actual next token).
    """
    if len(tokens) < 2:
        return 0.0, 0.0
    rows = chain.log_probs[chain.match_stream(tokens)[:-1]]
    nexts = np.asarray(tokens[1:], dtype=np.int64)
    nll = float(-rows[np.arange(len(nexts)), nexts].mean())
    return nll, float(np.mean(rows.argmax(axis=1) == nexts))


def unigram_nll(chain: Chain, tokens: Sequence[int]) -> float:
    """Mean NLL in nats of each next token under the chain's unigram row."""
    if len(tokens) < 2:
        return 0.0
    nexts = np.asarray(tokens[1:], dtype=np.int64)
    return float(-chain.log_probs[0][nexts].mean())


def model_predictor(model: HegelModel) -> Callable[[Sequence[int]], LogProbs]:
    """Wrap the model as history -> next-token log-probabilities."""

    def predict(history: Sequence[int]) -> LogProbs:
        window = list(history[-(model.max_offset + 1) :])
        with torch.no_grad():
            out = model(torch.as_tensor([window], dtype=torch.long))[0, -1]
        result: LogProbs = out.double().numpy()
        return result

    return predict


def sample(
    predict: Callable[[Sequence[int]], LogProbs],
    prompt: Sequence[int],
    length: int,
    draws: npt.NDArray[np.float64],
    temperature: float = 1.0,
) -> List[int]:
    """
    Extend the prompt by `length` tokens.

    Token t is chosen by inverting the predicted CDF at draws[t]. Two
    predictors given the same draws stay in step for as long as their
    distributions agree, so a divergence in the text is a real difference.

    temperature divides the log-probabilities before sampling: below 1 the
    likely tokens are favoured more, and 1 samples the distribution as it is.
    """
    history = list(prompt)
    for step in range(length):
        scaled = predict(history) / temperature
        cdf = np.cumsum(np.exp(scaled - scaled.max()))
        token = int(np.searchsorted(cdf, draws[step] * cdf[-1], side="right"))
        history.append(min(token, len(cdf) - 1))
    return history[len(prompt) :]
