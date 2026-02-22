"""
Tiktoken Tokenizer

Wrapper around OpenAI's tiktoken (GPT-2 BPE tokenizer) that matches
the interface expected by the AGI2 training and generation pipeline.
"""

from typing import Dict, List

import tiktoken


class TiktokenTokenizer:
    """
    GPT-2 BPE tokenizer via tiktoken.

    Uses the 'gpt2' encoding which has ~50,257 subword tokens.
    No fitting required — the vocabulary is pre-built.
    """

    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("gpt2")
        self._eos_token = "<|endoftext|>"
        self._eos_id = self._enc.encode(self._eos_token, allowed_special="all")[0]

        # Build vocab dict for compatibility (encode each token to verify roundtrip)
        self.vocab: Dict[str, int] = {"<EOS>": self._eos_id}
        self.vocab_size: int = self._enc.n_vocab

    def fit(self, texts: List[str]) -> None:
        """No-op. Tiktoken vocabulary is pre-built."""
        pass

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using GPT-2 BPE."""
        return self._enc.encode(text, allowed_special="all")

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self._enc.decode(token_ids)
