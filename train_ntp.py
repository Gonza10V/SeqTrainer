from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seqtrainer.genome_ntp.model import MACMemoryNTP, MacNTPConfig
from seqtrainer.genome_ntp.train_eval import load_datasets, make_loader, run_epoch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed/ecoli_mg1655")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="runs/ntp_smoke")
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, _, md = load_datasets(Path(args.processed), args.chunk_len, args.stride)
    train_loader, val_loader = make_loader(train_ds, args.batch_size), make_loader(val_ds, args.batch_size)

    cfg = MacNTPConfig(chunk_len=args.chunk_len)
    model = MACMemoryNTP(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metrics = []
    for e in range(args.epochs):
        model.reset_memory(); tr = run_epoch(model, train_loader, opt)
        model.reset_memory(); va = run_epoch(model, val_loader, None)
        row = {"epoch": e + 1, "train": tr, "val": va}
        metrics.append(row)
        print(json.dumps(row))
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "data_metadata": md}, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
