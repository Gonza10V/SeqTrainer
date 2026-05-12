from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GenomeWindowDataset


def bits_per_base(loss: float) -> float:
    return float(loss / np.log(2))


def run_epoch(model, loader, optim=None, device="cpu"):
    is_train = optim is not None
    model.train(is_train)
    losses, accs, surprises, mem_norms = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x, y=y, update_memory=True)
        loss = out["loss"]
        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        pred = out["logits"].argmax(-1)
        acc = (pred == y).float().mean().item()
        losses.append(loss.item()); accs.append(acc)
        if out["surprise"] is not None:
            surprises.append(out["surprise"])
        mem_norms.append(out["memory_norm"])
    return {
        "loss": float(np.mean(losses)),
        "perplexity": float(np.exp(np.mean(losses))),
        "bits_per_base": bits_per_base(float(np.mean(losses))),
        "accuracy": float(np.mean(accs)),
        "surprise": float(np.mean(surprises)) if surprises else None,
        "memory_norm": float(np.mean(mem_norms)),
    }


def confusion_matrix(model, loader, vocab_size: int, device="cpu"):
    cm = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device), y=None, update_memory=False)
            pred = out["logits"].argmax(-1).cpu().numpy().reshape(-1)
            tgt = y.numpy().reshape(-1)
            for t, p in zip(tgt, pred):
                cm[t, p] += 1
    return cm


def load_datasets(processed_dir: Path, chunk_len: int, stride: int):
    tokens = np.load(processed_dir / "tokens.npy").tolist()
    md = json.loads((processed_dir / "metadata.json").read_text())
    train = GenomeWindowDataset(tokens, tuple(md["splits"]["train"]), chunk_len, stride)
    val = GenomeWindowDataset(tokens, tuple(md["splits"]["val"]), chunk_len, stride)
    test = GenomeWindowDataset(tokens, tuple(md["splits"]["test"]), chunk_len, stride)
    return train, val, test, md


def make_loader(ds, batch_size: int):
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
