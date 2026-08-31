import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class WordTokenizer:
    """
    Deterministic word-level tokenizer for the layer tests.

    The SPM/LLD logic only cares about token-id sequences, so a real BPE tokenizer
    adds a model download without adding coverage. Ids are assigned from a growing
    vocabulary so identical words always map to identical ids.
    """

    def __init__(self):
        self._vocab = {}
        self._inv = {}

    def _id(self, word):
        if word not in self._vocab:
            idx = len(self._vocab)
            self._vocab[word] = idx
            self._inv[idx] = word
        return self._vocab[word]

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(self._inv.get(i, "<unk>") for i in ids)
