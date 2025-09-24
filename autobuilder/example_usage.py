"""Example script that stitches repository modules into a trainable network."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml

from .graph import build_model_from_config
from .registry import FileRegistration, ModuleRegistry
from .trainer import Trainer


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_registry(config: Dict[str, Any], base_dir: Path) -> ModuleRegistry:
    registry = ModuleRegistry()
    entries = []
    for entry in config.get("registry", []):
        file_path = Path(entry["file"])
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        entries.append(
            FileRegistration(
                file=file_path,
                class_name=entry.get("class_name"),
                alias=entry.get("alias"),
            )
        )
    registry.register_from_files(entries)
    return registry


def build_dataloaders(batch_size: int = 32, num_classes: int = 10) -> tuple[DataLoader, DataLoader]:
    """Creates a toy dataset so the training loop can be tested end-to-end."""

    num_samples = 512
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parent / "configs" / "example_classification.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to a YAML config describing the model/optimizer.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)

    registry = build_registry(config, base_dir=config_path.parent)
    model = build_model_from_config(config, registry)

    train_loader, val_loader = build_dataloaders()
    trainer = Trainer.from_config(model, config)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
