"""Portable client packets containing only sufficient statistics and a count."""

from pathlib import Path

import torch
from safetensors import safe_open

from parada.source import write_once_safetensors
from parada.sufficient_stats import RidgeStats, validate_packet

PACKET_FORMAT = "parada-ridge-v1"


def save_packet(path: Path, packet: RidgeStats) -> None:
    validate_packet(packet)
    write_once_safetensors(
        path,
        {
            "gram_upper": packet.gram_upper,
            "cross": packet.cross,
            "count": torch.tensor(packet.count, dtype=torch.int64),
        },
        {"format": PACKET_FORMAT, "client_id": str(packet.client_id)},
    )


def load_packet(path: Path) -> RidgeStats:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != PACKET_FORMAT or set(handle.keys()) != {
            "gram_upper",
            "cross",
            "count",
        }:
            raise ValueError("invalid sufficient-statistic packet format")
        try:
            client_id = int(metadata["client_id"])
        except (KeyError, ValueError) as exc:
            raise ValueError("packet client identity is missing or invalid") from exc
        count = handle.get_tensor("count")
        cross = handle.get_tensor("cross")
        if count.dtype != torch.int64 or count.ndim != 0 or cross.ndim != 2:
            raise ValueError("packet count must be an int64 scalar and cross must be rank two")
        packet = RidgeStats(
            client_id,
            int(count),
            cross.shape[0],
            cross.shape[1],
            handle.get_tensor("gram_upper"),
            cross,
        )
    validate_packet(packet)
    return packet
