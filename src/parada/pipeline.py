"""Validated tensor inputs for five-round logical-client simulation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from parada import source
from parada.federated import Client, build_prior, fit


def adapt_episode(
    model: torch.nn.Module,
    target_views: torch.Tensor,
    *,
    k: int,
    supports: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    seed: int = 42,
    device: str = "cpu",
):
    """Simulate ten clients; only residual/count packets enter aggregation.

    The sequence position is the client ID. This convenience orchestrator loads
    local support tensors; it is not a distributed server or network transport.
    """
    if type(k) is not int or not 0 <= k <= 10:
        raise ValueError("k must be an integer in 0..10")
    if (
        target_views.ndim != 3
        or target_views.shape[0] < 2
        or target_views.shape[1] != 3
        or target_views.shape[2] < 1
        or not target_views.is_floating_point()
        or not bool(torch.isfinite(target_views).all())
    ):
        raise ValueError("target_views must be finite floating [classes, 3, text_dim]")
    if k == 0 and supports is not None:
        raise ValueError("K=0 forbids support inputs")
    if k > 0 and (supports is None or len(supports) != 10):
        raise ValueError("K>0 requires exactly ten client support inputs")
    prior = build_prior(model.to(device), target_views.to(device)).cpu()
    if not bool(torch.isfinite(prior).all()) or bool((prior.norm(dim=1) == 0).any()):
        raise ValueError("source model must produce finite nonzero classifier rows")
    if k == 0:
        return prior, torch.zeros_like(prior), []
    assert supports is not None
    counts = torch.zeros(len(prior), dtype=torch.int64)
    for features, labels in supports:
        if (
            features.ndim != 2
            or features.shape[1] != prior.shape[1]
            or not features.is_floating_point()
            or not bool(torch.isfinite(features).all())
            or bool((features.float().norm(dim=1) == 0).any())
        ):
            raise ValueError("support features must be finite nonzero floating [samples, width]")
        if (
            labels.dtype != torch.int64
            or labels.shape != (len(features),)
            or bool(((labels < 0) | (labels >= len(prior))).any())
        ):
            raise ValueError("support labels must be int64 [samples] in shared class order")
        counts += torch.bincount(labels.cpu(), minlength=len(prior))
    if not bool((counts == k).all()):
        raise ValueError("global support must contain exactly K examples per class")
    class_ids = tuple(range(len(prior)))
    clients = [
        Client(i, features, labels, class_ids, prior, seed, device)
        for i, (features, labels) in enumerate(supports)
    ]
    return fit(clients, prior, class_ids, k)


def construct_classifier(
    model: torch.nn.Module,
    target_views: torch.Tensor,
    *,
    k: int,
    supports: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    return adapt_episode(model, target_views, k=k, supports=supports, seed=seed, device=device)[0]


def predict(features: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
    """Return scale-100 cosine logits in the shared target-class order."""
    return (
        source.LOGIT_SCALE
        * source.normalize_rows(features)
        @ F.normalize(classifier.detach().float().cpu(), dim=1, eps=source.EPS).T
    )
