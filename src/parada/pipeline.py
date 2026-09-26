"""MLP-only zero-shot prediction and one-shot analytic residual adaptation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from parada import source
from parada.sufficient_stats import RidgeStats, solve_prior_ridge


def construct_classifier(
    model: source.SourceMLP,
    target_views: torch.Tensor,
    *,
    k: int,
    packets: Sequence[RidgeStats] | None = None,
    regularization: float = 0.01,
    device: str = "cpu",
) -> torch.Tensor:
    """Construct class rows without accessing client examples or query data."""
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
    if k == 0 and packets is not None:
        raise ValueError("K=0 does not consume client statistics")
    if k > 0 and not packets:
        raise ValueError("K>0 requires client statistics")
    model.eval()
    prior = source.mlp_classifier(model, target_views, torch.device(device))
    if k == 0:
        return prior
    if packets is None:
        raise ValueError("K>0 requires client statistics")
    weights, _, record = solve_prior_ridge(prior, packets, regularization=regularization)
    if record["support_count"] != len(target_views) * k:
        raise ValueError("total support count must equal classes times K")
    return weights


def predict(features: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
    """Return scale-100 cosine logits in the shared target-class order."""
    return (
        source.LOGIT_SCALE
        * source.normalize_rows(features)
        @ F.normalize(classifier.detach().float().cpu(), dim=1, eps=source.EPS).T
    )
