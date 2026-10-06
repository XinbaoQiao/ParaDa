"""Bounded real-input check; does not train or claim benchmark accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from parada import prepare, source


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=prepare.CATALOG["models"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=prepare.DATASETS, default="cifar10")
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("verification output already exists")
    encoder = prepare.vision(args.profile, args.checkpoint, args.device)
    splits, index = prepare.dataset_index(args.dataset, args.data_root, encoder.transform)
    samples = []
    for split, dataset in splits.items():
        positions = sorted(set((0, len(dataset) // 2, len(dataset) - 1)))
        items = [dataset[position] for position in positions]
        images = torch.stack([item[1] for item in items])
        features = encoder.encode(images)
        direct = encoder.model(images.to(args.device)).float().cpu()
        bias = encoder._classifier.bias
        reconstructed = torch.nn.functional.linear(
            features,
            encoder.classification_head_rows(),
            None if bias is None else bias.detach().float().cpu(),
        )
        if not torch.isfinite(features).all():
            raise ValueError("real image features contain nonfinite values")
        torch.testing.assert_close(direct, reconstructed, rtol=1e-4, atol=1e-4)
        samples.append(
            {
                "split": split,
                "pool_size": len(dataset),
                "sample_ids": [item[0] for item in items],
                "feature_shape": list(features.shape),
                "feature_sha256": source.tensor_sha256(features),
                "classifier_max_abs_error": float((direct - reconstructed).abs().max()),
            }
        )
    del encoder
    clip, text_model = prepare.text_encoder(args.clip_checkpoint, args.device)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    if isinstance(descriptions, dict):
        descriptions = descriptions["descriptions"]
    if [row["class_name"] for row in descriptions] != index["class_names"]:
        raise ValueError("description classes differ from the real dataset")
    texts = [
        f"A photo of a {row['class_name']}: {row[key]}"
        for row in descriptions
        for key in ("appearance", "functionality", "environment")
    ]
    target = prepare.encode_text(clip, text_model, texts, args.device)
    normalized = prepare.encode_text(
        clip, text_model, index["class_names"], args.device, normalize=True
    )
    if target.shape != (len(descriptions) * 3, 512) or not torch.isfinite(target).all():
        raise ValueError("invalid real target text embeddings")
    torch.testing.assert_close(
        normalized.norm(dim=-1), torch.ones(len(descriptions)), rtol=2e-3, atol=2e-3
    )
    report = {
        "status": "passed",
        "claim_tier": "real-input-diagnostic",
        "profile": args.profile,
        "vision_sha256": prepare.CATALOG["models"][args.profile]["sha256"],
        "text_sha256": prepare.CATALOG["text"]["sha256"],
        "description_sha256": source.file_sha256(args.descriptions),
        "dataset": args.dataset,
        "index_sha256": index["index_sha256"],
        "device": args.device,
        "environment": prepare.environment(),
        "samples": samples,
        "target_shape": list(target.shape),
        "target_sha256": source.tensor_sha256(target),
        "full_extraction": False,
        "source_training": False,
        "benchmark_evaluation": False,
    }
    source.write_once_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
