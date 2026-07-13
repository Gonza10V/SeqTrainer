"""Command wrapper for official iPro-MP inference.

SeqTrainer does not vendor the upstream iPro-MP model code. This module gives
the Alpine workflow a stable package entrypoint that can invoke a user-provided
official inference script and then checks that the expected prediction CSV was
created.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dnabert-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--species-id", type=int, default=10)
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    script = os.environ.get("IPROMP_OFFICIAL_INFERENCE_SCRIPT")
    if not script:
        raise SystemExit(
            "Set IPROMP_OFFICIAL_INFERENCE_SCRIPT to the official iPro-MP inference script. "
            "SeqTrainer prepares FASTA inputs and evaluates prediction CSVs, but it does not "
            "vendor the upstream iPro-MP inference implementation."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        script,
        "--input-fasta",
        str(args.input_fasta),
        "--output-csv",
        str(args.output_csv),
        "--dnabert-dir",
        str(args.dnabert_dir),
        "--model-dir",
        str(args.model_dir),
        "--species-id",
        str(args.species_id),
        "--kmer-size",
        str(args.kmer_size),
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    subprocess.run(command, check=True)
    if not args.output_csv.exists():
        raise SystemExit(f"Official iPro-MP command completed but did not write {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
