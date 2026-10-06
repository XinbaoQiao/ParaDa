from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
from timm.data.config import resolve_data_config, resolve_model_data_config
from timm.data.transforms_factory import create_transform
from timm.models import load_checkpoint
from torch import nn
from torchvision import transforms

from parada._inputs.hashing import canonical_sha256, file_sha256


def _last_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    linear_layers = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if not linear_layers:
        raise ValueError("the supervised classifier does not expose a linear head")
    return linear_layers[-1]


def _capture_classifier_input(
    model: nn.Module, classifier: nn.Linear, backbone_features: torch.Tensor
) -> torch.Tensor:
    """Return the exact input consumed by a nested final linear classifier.

    Some timm ImageNet-21K models expose an MLP classifier whose final linear
    layer consumes a wider hidden representation than ``forward_head(...,
    pre_logits=True)`` returns.  A temporary pre-hook observes that final
    layer's input while preserving timm's own head ordering and eval behavior.
    """
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, args: tuple[object, ...]) -> None:
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise ValueError("timm classifier received an unsupported input signature")
        captured.append(args[0])

    handle = classifier.register_forward_pre_hook(capture)
    try:
        forward_head = getattr(model, "forward_head", None)
        if not callable(forward_head):
            raise ValueError("timm model does not expose forward_head")
        forward_head(backbone_features, pre_logits=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError(
            "timm classifier input capture must observe exactly one final-linear invocation"
        )
    return captured[0]


def _checkpoint_weights_identifier(
    checkpoint_path: str | Path, expected_sha256: str | None = None
) -> str:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ValueError(f"timm checkpoint is not an available file: {path}")
    actual = file_sha256(path)
    if expected_sha256 is not None:
        expected = expected_sha256.lower()
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError("timm expected_sha256 must be a 64-character hexadecimal digest")
        if actual != expected:
            raise ValueError(
                f"timm checkpoint SHA-256 mismatch: expected {expected}, observed {actual}"
            )
    return f"timm-checkpoint/{actual}"


def official_reference_transform() -> tuple[Any, dict[str, Any]]:
    """Return the pinned repository's BEiT evaluation preprocessing contract."""
    record: dict[str, Any] = {
        "profile": "official_code_13a2240",
        "resize": 256,
        "resize_interpolation": "torchvision_default_bilinear",
        "center_crop": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=record["mean"], std=record["std"]),
        ]
    )
    return (transform, record)


_BEIT_ALLOWED_UNEXPECTED_CHECKPOINT_KEYS = frozenset(
    f"blocks.{index}.attn.relative_position_index" for index in range(12)
)


def _load_explicit_timm_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a local checkpoint and fail closed on unaccounted key differences."""
    incompatible = load_checkpoint(model, str(checkpoint_path), strict=False)
    missing_keys = sorted(getattr(incompatible, "missing_keys", ()))
    unexpected_keys = sorted(getattr(incompatible, "unexpected_keys", ()))
    disallowed_unexpected = sorted(
        set(unexpected_keys).difference(_BEIT_ALLOWED_UNEXPECTED_CHECKPOINT_KEYS)
    )
    if missing_keys or disallowed_unexpected:
        raise ValueError(
            f"timm checkpoint is incompatible with the "
            f"instantiated model: missing_keys={missing_keys}, "
            f"disallowed_unexpected_keys={disallowed_unexpected}"
        )
    return {
        "loader": "timm.models.load_checkpoint",
        "strict": False,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "allowed_unexpected_keys": sorted(
            set(unexpected_keys).intersection(_BEIT_ALLOWED_UNEXPECTED_CHECKPOINT_KEYS)
        ),
        "validated": True,
    }


class TimmVisionEncoder:
    """Frozen timm model exposing pre-logit features and head rows from one checkpoint."""

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool = True,
        device: str = "cpu",
        checkpoint_path: str | Path | None = None,
        expected_sha256: str | None = None,
        preprocessing_profile: str = "timm",
    ) -> None:
        if preprocessing_profile not in {"timm", "official_13a2240", "official_code_13a2240"}:
            raise ValueError(f"unsupported timm preprocessing profile: {preprocessing_profile}")
        checkpoint_identifier: str | None = None
        kwargs: dict[str, Any] = {"pretrained": pretrained}
        if checkpoint_path is not None:
            checkpoint_identifier = _checkpoint_weights_identifier(
                checkpoint_path, expected_sha256=expected_sha256
            )
            kwargs["pretrained"] = False
        elif expected_sha256 is not None:
            raise ValueError("timm expected_sha256 requires an explicit checkpoint_path")
        model: Any = timm.create_model(model_name, **kwargs)
        if checkpoint_path is not None:
            self.checkpoint_compatibility = _load_explicit_timm_checkpoint(model, checkpoint_path)
        else:
            self.checkpoint_compatibility = {
                "loader": "timm.create_model",
                "strict": None,
                "missing_keys": [],
                "unexpected_keys": [],
                "allowed_unexpected_keys": [],
                "validated": pretrained,
            }
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model_name = model_name
        self.encoder_id = f"timm/{model_name}"
        if checkpoint_identifier is not None:
            self.weights_identifier = checkpoint_identifier
        elif pretrained:
            self.weights_identifier = (
                f"timm-weights/{canonical_sha256(getattr(model, 'pretrained_cfg', {}))}"
            )
        else:
            self.weights_identifier = f"timm-weights/{canonical_sha256(model.state_dict())}"
        classifier = _last_linear(model.get_classifier())
        self._classifier = classifier
        self.feature_dim = int(classifier.in_features)
        model_num_features = int(getattr(model, "num_features", self.feature_dim))
        self.feature_extraction_profile = (
            "timm_forward_head_pre_logits_v1"
            if model_num_features == self.feature_dim
            else "timm_final_linear_input_capture_v1"
        )
        self.classifier_path = str(getattr(model, "pretrained_cfg", {}).get("classifier", ""))
        if preprocessing_profile == "timm":
            try:
                data_config = resolve_model_data_config(model)
            except AttributeError:
                data_config = resolve_data_config(model.pretrained_cfg)
            self.preprocessing_identifier = (
                f"timm/{model_name}/{canonical_sha256(data_config)[:16]}"
            )
            self.transform = create_transform(**data_config, is_training=False)
        else:
            self.transform, preprocessing_record = official_reference_transform()
            self.preprocessing_identifier = (
                "official-code/13a2240/" + canonical_sha256(preprocessing_record)[:16]
            )

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(images.to(self.device))
        if self.feature_extraction_profile == "timm_forward_head_pre_logits_v1":
            pre_logits = self.model.forward_head(features, pre_logits=True)
        else:
            pre_logits = _capture_classifier_input(self.model, self._classifier, features)
        if pre_logits.ndim != 2 or pre_logits.shape[1] != self.feature_dim:
            raise ValueError("vision model returned incompatible pre-logit features")
        return pre_logits.to(torch.float32).cpu()

    def classification_head_rows(self) -> torch.Tensor:
        return self._classifier.weight.detach().to(torch.float32).cpu().clone()
