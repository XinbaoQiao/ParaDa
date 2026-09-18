"""Source-only centered ridge and local residual correction (Appendix B.2)."""
from __future__ import annotations
import hashlib
from collections.abc import Sequence
import torch
import torch.nn.functional as F

EPS = 1.0e-12


NEIGHBOR_COUNT = 16


TEMPERATURE = 0.1


RIDGE_RATIO = 0.01


class MethodError(RuntimeError):
    """Raised when frozen mathematics or tensor boundaries change."""


def _row_bytes(value: torch.Tensor) -> bytes:
    return value.detach().to(dtype=torch.float64, device="cpu").contiguous().numpy().tobytes()


def _anchor_id(value: torch.Tensor) -> str:
    return hashlib.sha256(_row_bytes(value)).hexdigest()


def normalize_rows(value: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    matrix = value.detach().to(dtype=dtype, device="cpu").contiguous()
    if matrix.ndim != 2 or not matrix.numel() or not bool(torch.isfinite(matrix).all()):
        raise MethodError("expected one non-empty finite rank-two matrix")
    if bool((torch.linalg.vector_norm(matrix, dim=1) <= EPS).any()):
        raise MethodError("matrix contains a zero-norm row")
    return F.normalize(matrix, dim=1, eps=EPS).contiguous()


def build_source_dictionary(
    source_text: torch.Tensor, source_visual: torch.Tensor, source_ids: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    text = normalize_rows(source_text, dtype=torch.float64)
    visual = normalize_rows(source_visual, dtype=torch.float64)
    if text.shape[0] != visual.shape[0] or len(source_ids) != text.shape[0]:
        raise MethodError("source text, visual rows and IDs are not aligned")
    if len(set(source_ids)) != len(source_ids):
        raise MethodError("source IDs must be unique and row aligned")
    groups: dict[bytes, list[int]] = {}
    for index, row in enumerate(text):
        groups.setdefault(row.contiguous().numpy().tobytes(), []).append(index)
    records: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for indices in groups.values():
        # The frozen source dictionary uses original row order keys
        # ``source:{zero-padded-index}``; within a duplicate semantic group
        # this is exactly ascending input row order.
        ordered = sorted(indices)
        semantic = text[ordered[0]]
        if len(ordered) == 1:
            visual_row = visual[ordered[0]].clone().contiguous()
        else:
            visual_row = normalize_rows(
                visual[torch.tensor(ordered)].mean(dim=0, keepdim=True), dtype=torch.float64
            )[0]
        records.append((_anchor_id(semantic), semantic, visual_row))
    records.sort(key=lambda item: item[0])
    return (
        torch.stack([item[1] for item in records]).contiguous(),
        torch.stack([item[2] for item in records]).contiguous(),
        tuple(item[0] for item in records),
    )


def _stable_neighbors(
    queries: torch.Tensor, anchors: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    query = normalize_rows(queries, dtype=torch.float64)
    source = normalize_rows(anchors, dtype=torch.float64)
    if query.shape[1] != source.shape[1]:
        raise MethodError("query and source semantic widths differ")
    if count < 1 or count > source.shape[0]:
        raise MethodError("neighbor count exceeds source dictionary")
    scores = query @ source.T
    anchor_id_to_index = {_anchor_id(row): index for index, row in enumerate(source)}
    for query_index, row in enumerate(query):
        exact_index = anchor_id_to_index.get(_anchor_id(row))
        if exact_index is not None:
            scores[query_index, exact_index] = 1.0
    indices = torch.arange(source.shape[0], dtype=torch.int64).expand(query.shape[0], -1)
    id_order = torch.argsort(indices, dim=1, stable=True)
    scores = torch.gather(scores, 1, id_order)
    indices = torch.gather(indices, 1, id_order)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]
    return torch.gather(scores, 1, order).contiguous(), torch.gather(indices, 1, order).contiguous()


def residual_kernel_classifier(
    source_text: torch.Tensor, source_visual: torch.Tensor, target_text_views: torch.Tensor
) -> torch.Tensor:
    if (
        source_text.ndim != 2
        or source_text.shape[1] < 1
        or target_text_views.ndim != 3
        or target_text_views.shape[1] != 3
        or target_text_views.shape[2] != source_text.shape[1]
    ):
        raise MethodError("target text views must match the source semantic width")
    if source_visual.ndim != 2 or source_visual.shape[1] < 1:
        raise MethodError("source visual classifiers must have a positive visual width")
    semantics = normalize_rows(source_text, dtype=torch.float64)
    visuals = normalize_rows(source_visual, dtype=torch.float64)
    semantic_mean = semantics.mean(dim=0)
    visual_mean = visuals.mean(dim=0)
    centered_x = semantics - semantic_mean
    centered_y = visuals - visual_mean
    gram = centered_x.T @ centered_x
    ridge_scale = torch.trace(gram) / float(gram.shape[0])
    if not bool(torch.isfinite(ridge_scale)) or float(ridge_scale) <= EPS:
        raise MethodError("source semantic ridge scale is degenerate")
    ridge_lambda = RIDGE_RATIO * float(ridge_scale)
    ridge_map = torch.cholesky_solve(
        centered_x.T @ centered_y,
        torch.linalg.cholesky(gram + ridge_lambda * torch.eye(gram.shape[0], dtype=torch.float64)),
    )
    fitted = visual_mean + centered_x @ ridge_map
    residuals = visuals - fitted
    queries = normalize_rows(
        target_text_views.detach().to(torch.float64).reshape(-1, target_text_views.shape[-1]),
        dtype=torch.float64,
    )
    scores, indices = _stable_neighbors(queries, semantics, NEIGHBOR_COUNT)
    shifted = (scores - scores.max(dim=1, keepdim=True).values) / TEMPERATURE
    weights = torch.softmax(shifted, dim=1)
    correction = torch.sum(weights[:, :, None] * residuals[indices], dim=1)
    base = visual_mean + (queries - semantic_mean) @ ridge_map
    views = normalize_rows(base + correction, dtype=torch.float64).reshape(
        target_text_views.shape[0], 3, visuals.shape[1]
    )
    return normalize_rows(views.mean(dim=1), dtype=torch.float64).to(torch.float32).contiguous()
