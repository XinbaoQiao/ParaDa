"""Portable composition of the existing numerical operators."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from parada import correction, source


def construct_classifier(
    model: source.SourceMLP,
    source_text: torch.Tensor,
    source_visual: torch.Tensor,
    target_views: torch.Tensor,
    *,
    k: int,
    support_features: torch.Tensor | None = None,
    support_labels: torch.Tensor | None = None,
    seed: int = 42,
    device: str = "cpu",
    rho_k0: float = 0.4,
    rho_positive: float = 1.0,
) -> torch.Tensor:
    """Construct rows in target_views order; queries never enter fitting.

    Defaults preserve the current shared-source implementation coefficients.
    The manuscript profile explicitly supplies rho_k0=0.5 (see ALIGNMENT.md).
    """
    if k not in range(11):
        raise ValueError("k must be an integer in 0..10")
    if not 0.0 <= rho_k0 <= 1.0 or not 0.0 <= rho_positive <= 1.0:
        raise ValueError("residual coefficients must be in [0, 1]")
    if target_views.ndim != 3 or target_views.shape[1] != 3:
        raise ValueError("target_views must have shape [classes, 3, text_dim]")
    if not bool(torch.isfinite(target_views).all()):
        raise ValueError("target_views must be finite")
    model.eval()
    prior = source.mlp_classifier(model, target_views, torch.device(device))
    if k == 0:
        if support_features is not None or support_labels is not None:
            raise ValueError("K=0 does not consume labeled support")
        text, visual, _ = correction.build_source_dictionary(
            source_text, source_visual, tuple(str(i) for i in range(len(source_text)))
        )
        residual_prior = correction.residual_kernel_classifier(text, visual, target_views)
        left = correction.normalize_rows(prior, dtype=torch.float64)
        right = correction.normalize_rows(residual_prior, dtype=torch.float64)
        return correction.normalize_rows(
            (1.0 - rho_k0) * left + rho_k0 * right, dtype=torch.float64
        ).float().contiguous()
    if support_features is None or support_labels is None:
        raise ValueError("K>0 requires support features and labels")
    if support_labels.dtype != torch.int64:
        raise ValueError("support labels must be int64")
    class_count = len(target_views)
    labels = support_labels.detach().cpu()
    if labels.ndim != 1 or labels.numel() != class_count * k:
        raise ValueError("support must contain exactly K rows for every selected class")
    if bool(((labels < 0) | (labels >= class_count)).any()):
        raise ValueError("support labels must index the selected target class order")
    if not bool((torch.bincount(labels, minlength=class_count) == k).all()):
        raise ValueError("support must contain exactly K rows for every selected class")
    rows, _ = source.fit_taskres_grid(
        prior, support_features, support_labels, k=k, rhos=(rho_positive,),
        seed=seed, device=torch.device(device),
    )
    return rows[0]


def predict(features: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
    """Return cosine logits; argmax gives indices in the shared class order."""
    return source.LOGIT_SCALE * source.normalize_rows(features) @ F.normalize(
        classifier.detach().float().cpu(), dim=1, eps=source.EPS
    ).T
