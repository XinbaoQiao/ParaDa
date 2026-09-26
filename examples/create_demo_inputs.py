"""Create small random tensors for the README example, not benchmark data."""

from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    root = Path("data/demo")
    generator = torch.Generator().manual_seed(18)
    tensors = {
        "source": {
            "source_text": torch.randn(20, 5, generator=generator),
            "source_visual": torch.randn(20, 4, generator=generator),
        },
        "target": {"target_views": torch.randn(3, 3, 5, generator=generator)},
        "support": {
            "support_features": torch.randn(3, 4, generator=generator),
            "support_labels": torch.arange(3, dtype=torch.int64),
        },
        "query": {"features": torch.randn(6, 4, generator=generator)},
    }
    support = tensors.pop("support")
    for client_id, indices in enumerate(([0, 1], [2], [])):
        index = torch.tensor(indices, dtype=torch.int64)
        tensors[f"client-{client_id}"] = {key: value[index] for key, value in support.items()}
    paths = {name: root / f"{name}.safetensors" for name in tensors}
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError("Demo inputs already exist; existing files were left untouched")
    root.mkdir(parents=True, exist_ok=True)
    for name, values in tensors.items():
        save_file(values, str(paths[name]))
    print("Created synthetic inputs in data/demo (not benchmark data)")


if __name__ == "__main__":
    main()
