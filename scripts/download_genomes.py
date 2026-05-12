#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from urllib.request import urlretrieve

from seqtrainer.genome_ntp.data import file_sha256


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--accession", default="NC_000913.3")
    p.add_argument("--out", default="data/raw/ecoli_mg1655")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = shutil.which("datasets")
    source = "ncbi_nuccore"
    fasta = out / f"{args.accession}.fna"
    if datasets:
        cmd = ["datasets", "download", "genome", "accession", "GCF_000005845.2", "--include", "genome,gff3,gbff", "--filename", str(out / "ncbi_dataset.zip")]
        subprocess.run(cmd, check=True)
        source = "ncbi_datasets_cli"
    if not fasta.exists():
        url = f"https://www.ncbi.nlm.nih.gov/search/api/sequence/{args.accession}/?report=fasta&format=text"
        urlretrieve(url, fasta)

    meta = {
        "accession": args.accession,
        "source": source,
        "download_date": str(date.today()),
        "file": str(fasta),
        "sha256": file_sha256(fasta),
    }
    (out / "download_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
