"""Lightweight training loop for dynamically assembled models."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    epochs: int = 1
    device: str = "auto"
    log_every: int = 10
    mixed_precision: bool = False


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.device = self._select_device(self.config.device)
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, model: torch.nn.Module, config: Mapping[str, Any]) -> "Trainer":
        trainer_cfg = TrainerConfig(**config.get("trainer", {}))
        loss_cfg = config["loss"]
        loss_fn = _instantiate_callable(loss_cfg["target"], loss_cfg.get("params", {}))

        opt_cfg = config["optimizer"]
        optimizer_cls = _instantiate_type(opt_cfg["target"])
        optimizer = optimizer_cls(model.parameters(), **opt_cfg.get("params", {}))

        scheduler_cfg = config.get("scheduler")
        scheduler = None
        if scheduler_cfg:
            scheduler_cls = _instantiate_type(scheduler_cfg["target"])
            scheduler = scheduler_cls(optimizer, **scheduler_cfg.get("params", {}))

        return cls(model, loss_fn, optimizer, scheduler, trainer_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = None
            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, training=False)
            message = f"epoch {epoch:03d} | train_loss={train_loss:.4f}"
            if val_loss is not None:
                message += f" | val_loss={val_loss:.4f}"
            print(message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, *, training: bool) -> float:
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(loader, start=1):
            inputs, target = self._split_batch(batch)
            inputs = self._move_to_device(inputs)
            target = target.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                output = self._forward(inputs)
                loss = self.loss_fn(output, target)
            if training:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            if training and step % self.config.log_every == 0:
                print(f"  step {step:04d} | loss={loss.item():.4f}")
        return total_loss / max(num_batches, 1)

    def _forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, Mapping):
            return self.model(**inputs)
        if isinstance(inputs, Sequence) and not isinstance(inputs, (torch.Tensor, str)):
            return self.model(*inputs)
        return self.model(inputs)

    def _move_to_device(self, inputs: Any) -> Any:
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        if isinstance(inputs, Mapping):
            return {key: self._move_to_device(value) for key, value in inputs.items()}
        if isinstance(inputs, Sequence):
            return type(inputs)(self._move_to_device(value) for value in inputs)  # type: ignore[call-arg]
        return inputs

    def _split_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        if isinstance(batch, Mapping):
            if "target" in batch:
                target = batch["target"]
                inputs = {k: v for k, v in batch.items() if k != "target"}
                return inputs, target
            if "y" in batch:
                target = batch["y"]
                inputs = {k: v for k, v in batch.items() if k != "y"}
                return inputs, target
            raise KeyError("Mapping batches must contain a 'target' or 'y' key")
        if isinstance(batch, Sequence) and not isinstance(batch, (torch.Tensor, str)):
            if len(batch) != 2:
                raise ValueError("Sequence batches must have length 2: (inputs, target)")
            return batch[0], batch[1]
        raise TypeError(
            "Unsupported batch structure. Provide (inputs, target) tuples or mappings "
            "with 'target'/'y' entries."
        )

    def _select_device(self, preference: str) -> torch.device:
        if preference == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():  # pragma: no cover - mac specific
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(preference)


def _instantiate_type(target: str) -> type:
    module_path, _, attr = target.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _instantiate_callable(target: str, params: Mapping[str, Any]) -> Any:
    cls = _instantiate_type(target)
    return cls(**params)


__all__ = ["Trainer", "TrainerConfig"]
