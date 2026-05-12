from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from seqtrainer.genome_ntp.model import MACMemoryNTP, MacNTPConfig
from seqtrainer.genome_ntp.train_eval import confusion_matrix, load_datasets, make_loader, run_epoch


def markov_baseline(tokens: list[int], vocab_size: int):
    mat = np.ones((vocab_size, vocab_size))
    for a, b in zip(tokens[:-1], tokens[1:]):
        mat[a, b] += 1
    return mat / mat.sum(axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed/ecoli_mg1655")
    p.add_argument("--checkpoint", default="runs/ntp_smoke/checkpoint.pt")
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = MacNTPConfig(**ckpt["config"])
    model = MACMemoryNTP(cfg); model.load_state_dict(ckpt["model"])
    _, _, test_ds, _ = load_datasets(Path(args.processed), args.chunk_len, args.stride)
    test_loader = make_loader(test_ds, 16)
    model.reset_memory(); metrics = run_epoch(model, test_loader, None)
    cm = confusion_matrix(model, test_loader, cfg.vocab_size)

    tok = np.load(Path(args.processed) / "tokens.npy").tolist()
    m = markov_baseline(tok, cfg.vocab_size)
    out = {
        "test": metrics,
        "confusion_matrix": cm.tolist(),
        "baseline_markov": {"transition_shape": list(m.shape)},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
