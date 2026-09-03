"""
BPE Tokenizer

Corpus-trained byte-level BPE via the HuggingFace `tokenizers` library
(Rust-backed). Matches the interface expected by the AGI2 training and
generation pipeline: fit / encode / decode / vocab / vocab_size.

Byte-level BPE guarantees lossless round-tripping of arbitrary text: the
initial alphabet covers all 256 bytes, so there are no unknown characters.
"""

from typing import Dict, List, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    """
    Byte-level BPE tokenizer trained on the project corpus.

    Unlike the pre-built tiktoken GPT-2 vocabulary, merges are learned from
    the training text itself, so a small vocabulary compresses the corpus
    well and every token embedding receives real training signal.

    Args:
        vocab_size: Target vocabulary size, including the 256 byte-level
            base tokens and the <EOS> special token. The fitted vocabulary
            may come out slightly smaller if the corpus supports fewer
            merges.
    """

    EOS_TOKEN = "<EOS>"

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self._tokenizer: Optional[Tokenizer] = None

    def fit(self, texts: List[str]) -> None:
        """
        Train the BPE vocabulary on a list of texts.

        Args:
            texts: List of text strings to train on
        """
        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[self.EOS_TOKEN],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)

        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.vocab = {self.EOS_TOKEN: tokenizer.token_to_id(self.EOS_TOKEN)}

    def _require_fitted(self) -> Tokenizer:
        if self._tokenizer is None:
            raise RuntimeError("BPETokenizer must be fitted before use")
        return self._tokenizer

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using the trained BPE merges."""
        return list(self._require_fitted().encode(text).ids)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return str(self._require_fitted().decode(token_ids, skip_special_tokens=False))

    def save_vocab(self, filepath: str) -> None:
        """Save the trained tokenizer (vocab + merges) to a JSON file."""
        self._require_fitted().save(filepath)

    def load_vocab(self, filepath: str) -> None:
        """Load a trained tokenizer (vocab + merges) from a JSON file."""
        tokenizer = Tokenizer.from_file(filepath)
        self._tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.vocab = {self.EOS_TOKEN: tokenizer.token_to_id(self.EOS_TOKEN)}
