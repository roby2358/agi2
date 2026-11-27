"""Generate text from a compiled model, given a source string."""

from typing import Callable, Sequence

import numpy as np
from src.corpus import normalize
from src.pipeline import Compiled
from src.verify import LogProbs, model_predictor, sample


def respond(
    compiled: Compiled,
    source: str,
    length: int,
    rng: np.random.Generator,
    use_chain: bool = False,
    temperature: float = 1.0,
) -> str:
    """
    Continue a source string.

    The source is normalized like the corpus (whitespace runs become one
    space) and encoded with the corpus vocabulary. Tokens the chain never
    saw in that order simply back off to shorter contexts.

    Args:
        compiled: A finished compile run.
        source: Text to continue.
        length: Number of tokens to generate.
        rng: Source of randomness.
        use_chain: Sample from the chain instead of the transformer.
        temperature: Below 1 favours likely tokens; 1 samples as is.

    Returns:
        Only the generated continuation, decoded to text.

    Raises:
        ValueError: If the source is empty, since the transformer predicts
            from a position and an empty source has none.
    """
    prompt = list(compiled.tokenizer.encode(normalize(source)))
    if not prompt:
        raise ValueError("source must contain at least one token")
    predict: Callable[[Sequence[int]], LogProbs] = (
        compiled.chain.next_log_probs if use_chain else model_predictor(compiled.model)
    )
    generated = sample(predict, prompt, length, rng.random(length), temperature)
    whole = str(compiled.tokenizer.decode(prompt + generated))
    return whole[len(str(compiled.tokenizer.decode(prompt))) :]
