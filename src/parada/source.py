"""Reusable source MLP and checkpoint handling."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CANDIDATE_ID = "parada"

EPS = 1.0e-12


LOGIT_SCALE = 100.0


MLP_HIDDEN_DIM = 3072


MLP_EPOCHS = 500


MLP_BATCH_SIZE = 512


MLP_LEARNING_RATE = 0.005


class SelectionError(RuntimeError):
    """Raised when a frozen selection input or information boundary changes."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = canonical_bytes(
        {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}
    )
    return hashlib.sha256(
        header + b"\0" + tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SelectionError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(json.dumps(dict(value), indent=2, sort_keys=True).encode() + b"\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_once_safetensors(
    path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]
) -> None:
    if path.exists() or path.is_symlink():
        raise SelectionError(f"write-once artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
        str(temporary),
        metadata=dict(metadata),
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_rows(value: torch.Tensor) -> torch.Tensor:
    matrix = value.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if matrix.ndim != 2 or not matrix.numel() or not bool(torch.isfinite(matrix).all()):
        raise SelectionError("expected one nonempty finite rank-two matrix")
    if bool((torch.linalg.vector_norm(matrix, dim=1) <= EPS).any()):
        raise SelectionError("matrix contains a zero-norm row")
    return F.normalize(matrix, dim=1, eps=EPS).contiguous()


MLP_PROFILE = {
    "architecture": "Linear-GELU-Dropout-Linear-LayerNorm",
    "hidden_dim": MLP_HIDDEN_DIM,
    "input_mask_gate_probability": 0.3,
    "input_keep_probability": 0.5,
    "hidden_dropout_probability": 0.5,
    "epochs": MLP_EPOCHS,
    "batch_size": MLP_BATCH_SIZE,
    "optimizer": "Adam",
    "learning_rate": MLP_LEARNING_RATE,
    "weight_decay": 0.0,
    "scheduler": "CosineAnnealingLR",
    "loss": "CosineEmbeddingLoss",
}


MLP_PROFILE_SHA256 = canonical_sha256(MLP_PROFILE)


class SourceMLP(nn.Module):
    def __init__(self, text_dim: int, visual_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(text_dim, MLP_HIDDEN_DIM)
        self.w2 = nn.Linear(MLP_HIDDEN_DIM, visual_dim)
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        value = text.float()
        if self.training and torch.rand(1, device=value.device).item() < 0.3:
            value = value * (torch.rand_like(value) > 0.5).float()
        value = F.dropout(F.gelu(self.w1(value)), p=0.5, training=self.training)
        return self.norm(self.w2(value)).float()


def state_hash(model: nn.Module) -> str:
    return canonical_sha256(
        {name: tensor_sha256(value) for name, value in model.state_dict().items()}
    )


def source_cache_key(source_text: torch.Tensor, source_visual: torch.Tensor, seed: int) -> str:
    return canonical_sha256(
        {
            "source_text_tensor_sha256": tensor_sha256(source_text),
            "source_visual_tensor_sha256": tensor_sha256(source_visual),
            "mlp_profile_sha256": MLP_PROFILE_SHA256,
            "seed": int(seed),
        }
    )


def train_or_load_source_model(
    source_text: torch.Tensor,
    source_visual: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    output_root: Path,
) -> tuple[SourceMLP, dict[str, Any]]:
    checkpoint_path = output_root / "source-checkpoint.safetensors"
    receipt_path = output_root / "source-checkpoint.json"
    key = source_cache_key(source_text, source_visual, seed)
    model = SourceMLP(source_text.shape[1], source_visual.shape[1])
    if checkpoint_path.exists() or receipt_path.exists():
        if not checkpoint_path.is_file() or not receipt_path.is_file():
            raise SelectionError("source checkpoint is partially persisted")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("cache_key") != key
            or receipt.get("seed") != seed
            or receipt.get("profile_sha256") != MLP_PROFILE_SHA256
            or receipt.get("checkpoint_file_sha256") != file_sha256(checkpoint_path)
        ):
            raise SelectionError("persisted source checkpoint receipt changed")
        values = load_file(str(checkpoint_path), device="cpu")
        model.load_state_dict(values, strict=True)
        if receipt.get("state_sha256") != state_hash(model):
            raise SelectionError("persisted source checkpoint state changed")
        return model.to(device).eval(), {**receipt, "execution": "reused"}

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    text = source_text.detach().to(torch.float32).cpu().contiguous()
    visual = normalize_rows(source_visual)
    loader = DataLoader(
        TensorDataset(text, visual),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MLP_EPOCHS)
    criterion = nn.CosineEmbeddingLoss()
    history: list[float] = []
    started = time.perf_counter()
    for _ in range(MLP_EPOCHS):
        total = 0.0
        batches = 0
        for text_batch, visual_batch in loader:
            prediction = model(text_batch.to(device))
            target = visual_batch.to(device)
            loss = criterion(prediction, target, torch.ones(prediction.shape[0], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
        history.append(total / batches)
        scheduler.step()
    elapsed = time.perf_counter() - started
    model.eval()
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    write_once_safetensors(
        checkpoint_path,
        state,
        {"candidate_id": CANDIDATE_ID, "cache_key": key, "seed": str(seed)},
    )
    receipt = {
        "schema_version": 1,
        "kind": "parada_shared_source_mlp_checkpoint_v1",
        "status": "trained_once",
        "candidate_id": CANDIDATE_ID,
        "cache_key": key,
        "seed": int(seed),
        "profile": MLP_PROFILE,
        "profile_sha256": MLP_PROFILE_SHA256,
        "source_text_tensor_sha256": tensor_sha256(source_text),
        "source_visual_tensor_sha256": tensor_sha256(source_visual),
        "state_sha256": canonical_sha256(
            {name: tensor_sha256(value) for name, value in state.items()}
        ),
        "checkpoint_file_sha256": file_sha256(checkpoint_path),
        "loss_curve_sha256": canonical_sha256(history),
        "first_loss": history[0],
        "final_loss": history[-1],
        "elapsed_seconds": elapsed,
        "target_dataset_opened_during_fit": False,
        "target_support_opened_during_fit": False,
        "execution": "trained",
    }
    write_once_json(receipt_path, receipt)
    return model, receipt


@torch.no_grad()
def mlp_classifier(
    model: SourceMLP, target_views: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if target_views.ndim != 3 or target_views.shape[1] != 3:
        raise SelectionError("target text must have exactly three views per class")
    class_count = int(target_views.shape[0])
    mapped = model(target_views.float().reshape(-1, target_views.shape[-1]).to(device))
    mapped = F.normalize(mapped, dim=1, eps=EPS).reshape(class_count, 3, -1).mean(dim=1)
    return F.normalize(mapped, dim=1, eps=EPS).cpu().contiguous()
