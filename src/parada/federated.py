"""One-shot client-class prototype packets and server-only residual fitting."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

METHOD_ID = "parada-prototype"


def tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(
        str((value.dtype, tuple(value.shape))).encode() + value.numpy().tobytes()
    ).hexdigest()


def schedule(k: int) -> tuple[float, ...]:
    if k not in range(1, 11):
        raise ValueError("K must be in 1..10")
    total = min(75 + 25 * k, 200)
    return (1e-5,) + tuple(
        0.001 * (1 + math.cos(math.pi * i / (total - 2))) for i in range(total - 1)
    )


@dataclass(frozen=True)
class PrototypePacket:
    client_id: int
    class_indices: tuple[int, ...]
    counts: torch.Tensor
    means: torch.Tensor


def client_packet(
    client_id: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> PrototypePacket:
    """Accumulate locally in FP64 and communicate only FP32 means/counts."""
    if not 0 <= client_id < 10 or num_classes < 2:
        raise ValueError("client or class axis invalid")
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("support axes mismatch")
    if not torch.isfinite(features).all():
        raise ValueError("nonfinite support")
    if len(labels) and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("support label outside class axis")
    x = F.normalize(features.detach().cpu().double(), dim=1)
    y = labels.detach().cpu().long()
    active = tuple(int(v) for v in torch.unique(y, sorted=True).tolist())
    counts = torch.tensor([int((y == c).sum()) for c in active], dtype=torch.int64)
    means = (
        torch.stack([x[y == c].mean(0) for c in active]).float()
        if active
        else torch.empty((0, x.shape[1]), dtype=torch.float32)
    )
    return PrototypePacket(client_id, active, counts, means)


def aggregate_packets(
    packets: list[PrototypePacket],
    num_classes: int,
    dim: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Server has packets only; no per-example support or label interface."""
    if len(packets) != 10 or {p.client_id for p in packets} != set(range(10)):
        raise ValueError("exactly ten distinct client packets required")
    sums = torch.zeros((num_classes, dim), dtype=torch.float64)
    counts = torch.zeros(num_classes, dtype=torch.int64)
    active_rows = 0
    for packet in sorted(packets, key=lambda p: p.client_id):
        if len(set(packet.class_indices)) != len(packet.class_indices):
            raise ValueError("duplicate class in packet")
        if (
            packet.means.shape != (len(packet.class_indices), dim)
            or packet.means.dtype != torch.float32
        ):
            raise ValueError("invalid FP32 mean packet")
        if (
            packet.counts.shape != (len(packet.class_indices),)
            or packet.counts.dtype != torch.int64
        ):
            raise ValueError("invalid count packet")
        if not torch.isfinite(packet.means).all() or bool((packet.counts <= 0).any()):
            raise ValueError("nonfinite or empty class packet")
        for row, c in enumerate(packet.class_indices):
            if not 0 <= c < num_classes:
                raise ValueError("class outside shared axis")
            n = int(packet.counts[row])
            sums[c] += packet.means[row].double() * n
            counts[c] += n
            active_rows += 1
    if not torch.equal(counts, torch.full_like(counts, k)):
        raise ValueError("global per-class K count mismatch")
    means = (sums / counts[:, None]).float()
    audit = {
        "client_packets": 10,
        "active_client_class_rows": active_rows,
        "support_count": int(counts.sum()),
        "class_counts": counts.tolist(),
        "mean_upload_bytes": active_rows * dim * 4,
        "count_upload_bytes": active_rows * 8,
        "global_prototype_sha256": tensor_digest(means),
    }
    return means, counts, audit


def fit_server(
    prior: torch.Tensor,
    examples: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Full-batch Adam with identical W0 and schedule for both information arms."""
    if prior.ndim != 2 or prior.dtype != torch.float32:
        raise ValueError("FP32 prior required")
    if examples.ndim != 2 or examples.shape[1] != prior.shape[1]:
        raise ValueError("feature dimension mismatch")
    if labels.shape != (len(examples),) or weights.shape != (len(examples),):
        raise ValueError("training rows mismatch")
    if len(examples) == 0 or int(labels.min()) < 0 or int(labels.max()) >= len(prior):
        raise ValueError("label axis invalid")
    if not torch.isfinite(examples).all() or not torch.isfinite(prior).all():
        raise ValueError("nonfinite input")
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("invalid row weights")
    device = prior.device
    x, y, row_weights = (
        examples.detach().float().to(device),
        labels.long().to(device),
        weights.float().to(device),
    )
    delta = torch.nn.Parameter(torch.zeros_like(prior))
    optimizer = torch.optim.Adam([delta], lr=1e-5, weight_decay=0, foreach=False, fused=False)
    initial_state_entries = len(optimizer.state)
    first_loss = None
    final_loss = None
    rates = schedule(k)
    for rate in rates:
        optimizer.param_groups[0]["lr"] = rate
        logits = 100 * (x @ F.normalize(prior + delta, dim=1).T)
        loss = (
            F.cross_entropy(logits, y, reduction="none") * row_weights
        ).sum() / row_weights.sum()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    learned = delta.detach().cpu().float().contiguous()
    head = F.normalize(prior.detach().cpu().float() + learned, dim=1)
    audit = {
        "optimizer": "full_batch_adam",
        "steps": len(rates),
        "initial_state_entries": initial_state_entries,
        "initial_loss": first_loss,
        "final_step_loss": final_loss,
        "prior_sha256": tensor_digest(prior),
        "delta_sha256": tensor_digest(learned),
        "head_sha256": tensor_digest(head),
    }
    return head, learned, audit


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


def fit(packets: list[PrototypePacket], prior: torch.Tensor, k: int):
    """Fit from ten prototype packets only; raw support rows stay at clients."""
    if prior.ndim != 2 or prior.shape[0] < 2:
        raise ValueError("prior and class axis mismatch")
    means, counts, packet_audit = aggregate_packets(packets, len(prior), prior.shape[1], k)
    head, delta, server_audit = fit_server(
        prior.detach(), means, torch.arange(len(prior)), counts.float(), k
    )
    return (
        head,
        delta,
        {
            "method": METHOD_ID,
            "communication_rounds": 1,
            **packet_audit,
            "server_fit": server_audit,
        },
    )
