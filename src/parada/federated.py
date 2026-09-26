"""Logical-client five-round residual adaptation; server consumes packets only."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

METHOD_ID = "five_round_residual"


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(
        str((value.dtype, tuple(value.shape))).encode() + value.numpy().tobytes()
    ).hexdigest()


def schedule(k: int) -> tuple[float, ...]:
    if k not in range(1, 11):
        raise ValueError("K must be an integer in 1..10")
    total = min(75 + 25 * k, 200)
    return (1e-5,) + tuple(
        0.001 * (1 + math.cos(math.pi * i / (total - 2))) for i in range(total - 1)
    )


@torch.no_grad()
def build_prior(model, views: torch.Tensor) -> torch.Tensor:
    if views.ndim != 3 or views.shape[1] != 3:
        raise ValueError("exactly three text descriptions per class required")
    model.eval()
    model.requires_grad_(False)
    mapped = model(views.float().reshape(-1, views.shape[-1]))
    return F.normalize(
        F.normalize(mapped, dim=1).reshape(views.shape[0], 3, -1).mean(1), dim=1
    ).detach()


@dataclass(frozen=True)
class Packet:
    client_id: int
    count: int
    delta: torch.Tensor


def aggregate(packets: list[Packet], class_ids: tuple[int, ...], shape: torch.Size) -> torch.Tensor:
    """FP32 sample-weighted unnormalized residual; no data/labels in this API."""
    if len({p.client_id for p in packets}) != len(packets):
        raise ValueError("duplicate client")
    total = sum(p.count for p in packets)
    if total <= 0:
        raise ValueError("no support")
    result = torch.zeros(shape, dtype=torch.float32)
    for p in sorted(packets, key=lambda p: p.client_id):
        if p.count < 0 or p.delta.shape != shape or shape[0] != len(class_ids):
            raise ValueError("packet class axis, count, or shape mismatch")
        if p.delta.dtype != torch.float32 or not torch.isfinite(p.delta).all():
            raise ValueError("invalid communication tensor")
        if p.count:
            result.add_(p.delta.detach().cpu(), alpha=p.count / total)
    return result


class Client:
    """Owns support features/labels; query is absent from training interface."""

    def __init__(self, client_id, features, labels, class_ids, prior, seed, device):
        self.client_id = int(client_id)
        self.class_ids = tuple(class_ids)
        self.prior = prior.detach().float().to(device).clone()
        if (
            self.prior.ndim != 2
            or self.prior.shape[0] != len(self.class_ids)
            or len(set(self.class_ids)) != len(self.class_ids)
        ):
            raise ValueError("prior and class axis mismatch")
        self.x = F.normalize(features.detach().float().to(device), dim=1)
        self.y = labels.detach().long().to(device).clone()
        if self.x.ndim != 2 or self.y.shape != (len(self.x),) or self.x.shape[1] != prior.shape[1]:
            raise ValueError("client support axes mismatch")
        if len(self.y) and (int(self.y.min()) < 0 or int(self.y.max()) >= len(class_ids)):
            raise ValueError("client label outside shared class order")
        self.shuffle_seed = int(seed) + 1000003 * self.client_id
        self.generator = torch.Generator(device="cpu").manual_seed(self.shuffle_seed)

    def update(self, broadcast, rates):
        initial_hash = tensor_digest(broadcast)
        if not len(self.y):
            raise ValueError("empty clients do not receive or send delta payloads")
        delta = torch.nn.Parameter(broadcast.detach().to(self.prior.device).clone())
        optimizer = torch.optim.Adam(
            [delta], lr=rates[0], weight_decay=0, foreach=False, fused=False
        )
        assert not optimizer.state
        steps = 0
        for rate in rates:
            optimizer.param_groups[0]["lr"] = rate
            order = torch.randperm(len(self.y), generator=self.generator)
            for indices in order.split(256):
                indices = indices.to(self.x.device)
                weights = F.normalize(self.prior + delta, dim=1)
                loss = F.cross_entropy(100 * (self.x[indices] @ weights.T), self.y[indices])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
        value = delta.detach().cpu().float().contiguous()
        return Packet(self.client_id, len(self.y), value), {
            "count": len(self.y),
            "initial_delta": initial_hash,
            "optimizer_steps": steps,
            "optimizer_initial_state_entries": 0,
            "final_delta": tensor_digest(value),
        }


def fit(clients: list[Client], prior: torch.Tensor, class_ids: tuple[int, ...], k: int):
    """Five rounds, episode-local zero initialization and complete broadcast."""
    if len(clients) != 10 or {c.client_id for c in clients} != set(range(10)):
        raise ValueError("ten distinct logical clients required")
    rates = schedule(k)
    epochs = len(rates) // 5
    delta = torch.zeros_like(prior, dtype=torch.float32, device="cpu")
    frozen_prior_hash = tensor_digest(prior)
    if prior.ndim != 2 or prior.shape[0] != len(class_ids) or prior.dtype != torch.float32:
        raise ValueError("prior and class axis mismatch")
    for client in clients:
        if client.class_ids != class_ids or tensor_digest(client.prior) != frozen_prior_hash:
            raise ValueError("client prior or class axis differs from episode")
        client.generator.manual_seed(client.shuffle_seed)
    records = []
    for r in range(5):
        packets, audits = [], []
        for client in clients:
            if not len(client.y):
                audits.append(
                    {
                        "client_id": client.client_id,
                        "count": 0,
                        "initial_delta": None,
                        "optimizer_steps": 0,
                        "delta_payload_sent": False,
                    }
                )
                continue
            packet, audit = client.update(delta.clone(), rates[r * epochs : (r + 1) * epochs])
            packets.append(packet)
            audits.append({"client_id": client.client_id, **audit})
        before = tensor_digest(delta)
        delta = aggregate(packets, class_ids, prior.shape)
        active = sum(p.count > 0 for p in packets)
        records.append(
            {
                "round": r + 1,
                "broadcast_delta": before,
                "aggregate_delta": tensor_digest(delta),
                "first_lr": rates[r * epochs],
                "last_lr": rates[(r + 1) * epochs - 1],
                "clients": audits,
                "active_clients": active,
                "support_count": sum(p.count for p in packets),
                "upload_delta_bytes": active * delta.numel() * 4,
                "upload_count_bytes": active * 8,
                "download_delta_bytes": active * delta.numel() * 4,
            }
        )
    assert tensor_digest(prior) == frozen_prior_hash
    return F.normalize(prior.detach().cpu().float() + delta, dim=1), delta, records
