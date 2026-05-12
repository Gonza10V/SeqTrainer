from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DNATokenizer:
    pad_token: str = "<PAD>"

    def __post_init__(self) -> None:
        self.vocab = [self.pad_token, "A", "C", "G", "T", "N"]
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for i, t in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def normalize(self, seq: str) -> str:
        seq = (seq or "").upper()
        return "".join(ch if ch in {"A", "C", "G", "T", "N"} else "N" for ch in seq)

    def encode(self, seq: str) -> list[int]:
        return [self.stoi[ch] for ch in self.normalize(seq)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != self.stoi[self.pad_token])


def reverse_complement(seq: str) -> str:
    comp = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"})
    return seq.upper().translate(comp)[::-1]
