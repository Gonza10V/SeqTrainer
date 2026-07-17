from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GenomeEntry:
    name: str
    accession: str
    fasta_path: str
    group: str


@dataclass
class GenomeManifest:
    genomes: list[GenomeEntry]

    @classmethod
    def load(cls, path: Path) -> "GenomeManifest":
        raw = json.loads(path.read_text())
        return cls(genomes=[GenomeEntry(**g) for g in raw["genomes"]])

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps({"genomes": [asdict(g) for g in self.genomes]}, indent=2))
