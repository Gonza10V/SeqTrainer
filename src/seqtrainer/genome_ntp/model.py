from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MacNTPConfig:
    vocab_size: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    chunk_len: int = 256
    memory_slots: int = 8
    memory_decay: float = 0.95


class MACMemoryNTP(nn.Module):
    def __init__(self, cfg: MacNTPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.chunk_len + cfg.memory_slots, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.mem_update = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.Tanh(), nn.Linear(cfg.d_model, cfg.d_model))
        self.register_buffer("memory", torch.zeros(cfg.memory_slots, cfg.d_model))

    def reset_memory(self) -> None:
        self.memory.zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None, update_memory: bool = True) -> dict:
        b, t = x.shape
        tok = self.emb(x)
        mem = self.memory.unsqueeze(0).expand(b, -1, -1)
        h = torch.cat([mem, tok], dim=1)
        h = h + self.pos(torch.arange(h.shape[1], device=x.device)).unsqueeze(0)
        causal_mask = torch.triu(torch.full((h.shape[1], h.shape[1]), float("-inf"), device=x.device), diagonal=1)
        enc = self.enc(h, mask=causal_mask)
        tok_enc = enc[:, self.cfg.memory_slots :, :]
        logits = self.lm_head(tok_enc)
        out = {"logits": logits}
        probs = torch.softmax(logits.detach(), dim=-1)
        surprise = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            out["loss"] = loss
            pred_p = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            surprise = (1.0 - pred_p).mean().item()
        if update_memory:
            update_vec = self.mem_update(tok_enc.mean(dim=1)).mean(dim=0)
            decay = self.cfg.memory_decay
            with torch.no_grad():
                self.memory.mul_(decay).add_((1 - decay) * update_vec.unsqueeze(0))
        out["memory_norm"] = self.memory.norm().item()
        out["surprise"] = float(surprise) if surprise is not None else None
        return out
