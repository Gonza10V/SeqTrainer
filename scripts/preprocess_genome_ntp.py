#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from seqtrainer.genome_ntp.data import build_processed_from_fasta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", default="data/raw/ecoli_mg1655/NC_000913.3.fna")
    p.add_argument("--out", default="data/processed/ecoli_mg1655")
    p.add_argument("--accession", default="NC_000913.3")
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--buffer-bp", type=int, default=2048)
    p.add_argument("--include-reverse-complement", action="store_true")
    args = p.parse_args()
    build_processed_from_fasta(
        Path(args.fasta), Path(args.out), args.accession, args.include_reverse_complement, args.chunk_len, args.stride, args.buffer_bp
    )


if __name__ == "__main__":
    main()
