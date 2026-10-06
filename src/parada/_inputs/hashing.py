from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and (not isinstance(value, type)):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": tensor_sha256(value),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes, bytearray))):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        raise ValueError("canonical JSON does not permit NaN or infinity")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a value as stable UTF-8 JSON with canonical key ordering."""
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous CPU bytes."""
    value = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape)}
    )
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return sha256_bytes(header + b"\x00" + raw)
