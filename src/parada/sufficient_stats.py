"""One-shot federated ridge adaptation around a frozen classifier prior.

Clients upload only additive sufficient statistics. The server never receives
the client API's feature matrix or label vector. This is a simulation boundary,
not a privacy guarantee: sparse class sums can reveal individual samples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RidgeStats:
    client_id: int
    count: int
    width: int
    classes: int
    gram_upper: torch.Tensor
    cross: torch.Tensor

    @property
    def payload_bytes(self) -> int:
        return (
            self.gram_upper.numel() * self.gram_upper.element_size()
            + self.cross.numel() * self.cross.element_size()
            + 8
        )


def validate_packet(packet: RidgeStats) -> None:
    """Validate the tensor schema; this does not authenticate a client."""
    if any(
        type(v) is not int for v in (packet.client_id, packet.count, packet.width, packet.classes)
    ):
        raise ValueError("packet identifiers, count, and dimensions must be integers")
    if not (0 <= packet.client_id < 2**63 and 0 <= packet.count < 2**63):
        raise ValueError("packet identity and count must be nonnegative int64 values")
    if packet.width < 1 or packet.classes < 2:
        raise ValueError("packet dimensions are invalid")
    if packet.gram_upper.shape != (
        packet.width * (packet.width + 1) // 2,
    ) or packet.cross.shape != (packet.width, packet.classes):
        raise ValueError("packet tensor shapes differ from its dimensions")
    if (
        packet.gram_upper.dtype not in (torch.float32, torch.float64)
        or packet.cross.dtype != packet.gram_upper.dtype
    ):
        raise ValueError("packet tensors must share float32 or float64 precision")
    if not all(bool(torch.isfinite(t).all()) for t in (packet.gram_upper, packet.cross)):
        raise ValueError("packet values must be finite")
    if packet.count == 0 and any(
        bool(torch.count_nonzero(t)) for t in (packet.gram_upper, packet.cross)
    ):
        raise ValueError("empty client packets must contain zero statistics")


def client_statistics(
    client_id: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    classes: int,
    communication_dtype: torch.dtype = torch.float32,
) -> RidgeStats:
    """Compute float64 local sums and send one quantized stat packet."""

    if type(client_id) is not int or not 0 <= client_id < 2**63:
        raise ValueError("client identity must be a nonnegative int64 value")
    if type(classes) is not int or labels.dtype != torch.int64 or not features.is_floating_point():
        raise ValueError("classes must be an integer, features floating point, and labels int64")
    if communication_dtype not in (torch.float32, torch.float64):
        raise ValueError("statistics communication must use float32 or float64")
    if features.ndim != 2 or features.shape[1] < 1 or labels.shape != (features.shape[0],):
        raise ValueError("client feature/label shapes differ")
    if classes < 2 or not bool(torch.isfinite(features).all()):
        raise ValueError("class count or feature values are invalid")
    y = labels.detach().to(device="cpu", dtype=torch.int64)
    if y.numel() and (int(y.min()) < 0 or int(y.max()) >= classes):
        raise ValueError("client label is outside class axis")
    raw = features.detach().to(device="cpu", dtype=torch.float64)
    if bool((torch.linalg.vector_norm(raw, dim=1) <= 1e-12).any()):
        raise ValueError("support features must have nonzero rows")
    x = F.normalize(raw, dim=1)
    width = x.shape[1]
    gram = x.T @ x
    cross = torch.zeros((width, classes), dtype=torch.float64)
    if y.numel():
        cross.index_add_(1, y, x.T)
    upper = gram[torch.triu_indices(width, width).unbind()]
    return RidgeStats(
        client_id=int(client_id),
        count=int(y.numel()),
        width=width,
        classes=classes,
        gram_upper=upper.to(communication_dtype).contiguous(),
        cross=cross.to(communication_dtype).contiguous(),
    )


def solve_prior_ridge(
    prior: torch.Tensor,
    packets: Sequence[RidgeStats],
    *,
    regularization: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Solve min mean||X(W0+Delta)^T-Y||² + lambda||Delta||² exactly for received stats."""

    if regularization <= 0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    if prior.ndim != 2 or not prior.numel() or not bool(torch.isfinite(prior).all()):
        raise ValueError("prior must be finite class-by-feature weights")
    classes, width = prior.shape
    if classes < 2 or width < 1 or not prior.is_floating_point():
        raise ValueError("prior must be floating point with at least two classes")
    if bool((torch.linalg.vector_norm(prior.double(), dim=1) <= 1e-12).any()):
        raise ValueError("prior rows must be nonzero")
    for packet in packets:
        validate_packet(packet)
    if not packets or len({p.client_id for p in packets}) != len(packets):
        raise ValueError("packets must have unique client identities")
    size = width * (width + 1) // 2
    gram_upper = torch.zeros(size, dtype=torch.float64)
    cross = torch.zeros((width, classes), dtype=torch.float64)
    total = 0
    for packet in packets:
        if packet.cross.dtype != packets[0].cross.dtype:
            raise ValueError("all packets must use the same communication precision")
        if (
            packet.width != width
            or packet.classes != classes
            or packet.count < 0
            or packet.gram_upper.shape != (size,)
            or packet.cross.shape != (width, classes)
            or packet.gram_upper.dtype not in (torch.float32, torch.float64)
            or packet.cross.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(packet.gram_upper).all())
            or not bool(torch.isfinite(packet.cross).all())
        ):
            raise ValueError("stat packet does not match the frozen classifier axis")
        gram_upper.add_(packet.gram_upper.to(device="cpu", dtype=torch.float64))
        cross.add_(packet.cross.to(device="cpu", dtype=torch.float64))
        total += packet.count
    if total == 0:
        raise ValueError("no labeled support in client packets")
    indices = torch.triu_indices(width, width)
    gram = torch.zeros((width, width), dtype=torch.float64)
    gram[indices[0], indices[1]] = gram_upper
    gram = gram + gram.T - torch.diag(torch.diag(gram))
    w0 = F.normalize(prior.detach().to(device="cpu", dtype=torch.float64), dim=1).T
    left = gram + (total * regularization) * torch.eye(width, dtype=torch.float64)
    right = cross - gram @ w0
    delta = torch.linalg.solve(left, right)
    weights = F.normalize((w0 + delta).T, dim=1).to(torch.float32).contiguous()
    residual = delta.T.to(torch.float32).contiguous()
    equation_error = torch.linalg.vector_norm(left @ delta - right).item()
    return (
        weights,
        residual,
        {
            "method_id": "parada_one_shot_prior_ridge_v1",
            "objective": "mean_squared_onehot_score_plus_lambda_delta_squared",
            "regularization": regularization,
            "client_count": len(packets),
            "support_count": total,
            "uploads_per_client": 1,
            "uploaded_support_features_or_labels": False,
            "uploaded_payload": "symmetric_xtx_upper_and_xty_and_count",
            "communication_dtype": str(packets[0].cross.dtype),
            "total_upload_bytes": sum(p.payload_bytes for p in packets),
            "normal_equation_residual_l2": equation_error,
            "sparse_statistics_may_reveal_individual_samples": True,
        },
    )
