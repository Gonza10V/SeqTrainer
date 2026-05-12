from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import DNATokenizer, reverse_complement


def parse_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current = line[1:].split()[0]
                records[current] = []
            elif current is not None:
                records[current].append(line)
    return {k: "".join(v) for k, v in records.items()}


@dataclass
class IntervalSplit:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def interval_split(length: int, train_frac: float = 0.8, val_frac: float = 0.1, buffer_bp: int = 2048) -> IntervalSplit:
    train_end = int(length * train_frac)
    val_end = int(length * (train_frac + val_frac))
    train = (0, max(0, train_end - buffer_bp))
    val = (min(length, train_end + buffer_bp), max(0, val_end - buffer_bp))
    test = (min(length, val_end + buffer_bp), length)
    return IntervalSplit(train=train, val=val, test=test)


class GenomeWindowDataset(Dataset):
    def __init__(self, token_ids: list[int], interval: tuple[int, int], chunk_len: int, stride: int) -> None:
        self.tokens = token_ids
        self.start, self.end = interval
        self.chunk_len = chunk_len
        self.stride = stride
        self.window_starts = list(range(self.start, max(self.start, self.end - chunk_len - 1) + 1, stride))

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.window_starts[idx]
        x = self.tokens[s : s + self.chunk_len]
        y = self.tokens[s + 1 : s + self.chunk_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_processed_from_fasta(
    fasta_path: Path,
    out_dir: Path,
    accession: str,
    include_reverse_complement: bool,
    chunk_len: int,
    stride: int,
    buffer_bp: int,
) -> None:
    tok = DNATokenizer()
    records = parse_fasta(fasta_path)
    seq = tok.normalize(next(iter(records.values())))
    if include_reverse_complement:
        seq = seq + "N" + reverse_complement(seq)
    token_ids = tok.encode(seq)
    split = interval_split(len(token_ids), buffer_bp=buffer_bp)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tokens.npy", np.array(token_ids, dtype=np.int16))
    write_metadata(
        out_dir / "metadata.json",
        {
            "accession": accession,
            "source_fasta": str(fasta_path),
            "sha256": file_sha256(fasta_path),
            "sequence_length": len(token_ids),
            "preprocessing": {
                "include_reverse_complement": include_reverse_complement,
                "chunk_len": chunk_len,
                "stride": stride,
                "buffer_bp": buffer_bp,
            },
            "splits": {"train": split.train, "val": split.val, "test": split.test},
        },
    )
