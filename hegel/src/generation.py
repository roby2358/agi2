"""Generate text from a compiled model, given a source string."""

from typing import Any, Callable, Sequence

import numpy as np
from src.corpus import normalize
from src.verify import LogProbs, sample


def respond(
    tokenizer: Any,
    predict: Callable[[Sequence[int]], LogProbs],
    source: str,
    length: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> str:
    """
    Continue a source string.

    The source is normalized like the corpus (whitespace runs become one
    space) and encoded with the corpus vocabulary. Tokens the chain never
    saw in that order simply back off to shorter contexts.

    Args:
        tokenizer: The corpus tokenizer (encode / decode).
        predict: History in, next-token log-probabilities out: the
            transformer (verify.model_predictor) or chain.next_log_probs.
        source: Text to continue.
        length: Number of tokens to generate.
        rng: Source of randomness.
        temperature: Below 1 favours likely tokens; 1 samples as is.

    Returns:
        Only the generated continuation, decoded to text.

    Raises:
        ValueError: If the source is empty, since the transformer predicts
            from a position and an empty source has none.
    """
    prompt = list(tokenizer.encode(normalize(source)))
    if not prompt:
        raise ValueError("source must contain at least one token")
    generated = sample(predict, prompt, length, rng.random(length), temperature)
    whole = str(tokenizer.decode(prompt + generated))
    return whole[len(str(tokenizer.decode(prompt))) :]
