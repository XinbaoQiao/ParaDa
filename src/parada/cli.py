"""Tensor-input command line for source training, adaptation, and prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from parada import source
from parada.pipeline import construct_classifier, predict


def _source_inputs(path: Path) -> dict[str, torch.Tensor]:
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"source_text", "source_visual"}:
        raise ValueError("source file must contain source_text and source_visual only")
    text, visual = tensors["source_text"], tensors["source_visual"]
    if text.ndim != 2 or visual.ndim != 2 or len(text) != len(visual) or len(text) < 16:
        raise ValueError("source inputs require at least 16 aligned rows and positive widths")
    source.normalize_rows(text)
    source.normalize_rows(visual)
    return tensors


def _model(path: Path, tensors: dict[str, torch.Tensor], seed: int, device: str):
    receipt = json.loads((path / "source-checkpoint.json").read_text(encoding="utf-8"))
    if receipt.get("cache_key") != source.source_cache_key(
        tensors["source_text"], tensors["source_visual"], seed
    ):
        raise ValueError("checkpoint does not match the supplied source inputs and seed")
    weights = path / "source-checkpoint.safetensors"
    if receipt.get("checkpoint_file_sha256") != source.file_sha256(weights):
        raise ValueError("checkpoint file hash differs from its receipt")
    model = source.SourceMLP(tensors["source_text"].shape[1], tensors["source_visual"].shape[1])
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    if source.state_hash(model) != receipt.get("state_sha256"):
        raise ValueError("checkpoint state hash differs from its receipt")
    return model.to(device).eval()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="parada",
        description="ParaDa: Pre-trained Parameters as Data for Federated Few-Shot Learning",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train or reuse the source-only MLP")
    train.add_argument("--source", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    adapt = commands.add_parser("adapt", help="construct target rows and optionally fit support")
    adapt.add_argument("--source", type=Path, required=True)
    adapt.add_argument("--checkpoint", type=Path, required=True)
    adapt.add_argument("--target", type=Path, required=True)
    adapt.add_argument("--support", type=Path)
    adapt.add_argument("--k", type=int, choices=range(11), required=True)
    adapt.add_argument("--profile", choices=("current", "manuscript"), default="current")
    adapt.add_argument("--seed", type=int, default=42)
    adapt.add_argument("--device", default="cpu")
    adapt.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("predict", help="score frozen query features")
    evaluate.add_argument("--features", type=Path, required=True)
    evaluate.add_argument("--classifier", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        tensors = _source_inputs(args.source)
        _, receipt = source.train_or_load_source_model(
            **tensors, seed=args.seed, device=torch.device(args.device), output_root=args.checkpoint
        )
        print(json.dumps({"execution": receipt["execution"], "state_sha256": receipt["state_sha256"]}))
        return
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.command == "adapt":
        tensors = _source_inputs(args.source)
        target = load_file(str(args.target), device="cpu")
        if set(target) != {"target_views"}:
            raise ValueError("target file must contain target_views only")
        support = {} if args.support is None else load_file(str(args.support), device="cpu")
        if support and set(support) != {"support_features", "support_labels"}:
            raise ValueError("support file must contain support_features and support_labels only")
        model = _model(args.checkpoint, tensors, args.seed, args.device)
        classifier = construct_classifier(
            model, **tensors, **target, **support, k=args.k, seed=args.seed, device=args.device,
            rho_k0=0.4 if args.profile == "current" else 0.5, rho_positive=1.0,
        )
        source.write_once_safetensors(args.output, {"classifier": classifier}, metadata={
            "profile": args.profile, "k": str(args.k), "seed": str(args.seed),
            "source_checkpoint_sha256": source.file_sha256(args.checkpoint / "source-checkpoint.safetensors"),
            "source_inputs_sha256": source.file_sha256(args.source),
            "target_inputs_sha256": source.file_sha256(args.target),
            "support_inputs_sha256": "none" if args.support is None else source.file_sha256(args.support),
        })
    else:
        features = load_file(str(args.features), device="cpu")
        classifier = load_file(str(args.classifier), device="cpu")
        if set(features) != {"features"} or set(classifier) != {"classifier"}:
            raise ValueError("prediction inputs require features and classifier respectively")
        scores = predict(features["features"], classifier["classifier"])
        source.write_once_safetensors(
            args.output, {"logits": scores, "predictions": scores.argmax(dim=1)}, metadata={}
        )
    print(json.dumps({"output": str(args.output)}))
