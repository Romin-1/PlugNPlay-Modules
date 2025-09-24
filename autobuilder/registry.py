"""Module registry utilities.

The goal of the registry is to make it trivial to mix native PyTorch layers
with the modules that ship with this repository.  The class defined here keeps
track of ``nn.Module`` subclasses and exposes a uniform resolver that works
with both explicit registrations and fully qualified import paths.
"""
from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Type

import torch.nn as nn


@dataclass
class FileRegistration:
    """Describes how to load a module class from a Python file."""

    file: Path
    class_name: Optional[str] = None
    alias: Optional[str] = None


class ModuleRegistry:
    """Collects ``nn.Module`` classes that can be instantiated on demand.

    The registry supports three main workflows:

    1. ``register`` – push an already imported class into the registry.
    2. ``register_from_file`` – load a module from a Python file located in the
       repository and register all ``nn.Module`` subclasses that it exposes.
    3. ``resolve`` – lookup a class either by registry key or by fully qualified
       import path (e.g. ``torch.nn.Conv2d``).
    """

    def __init__(self) -> None:
        self._store: Dict[str, Type[nn.Module]] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register(self, name: str, cls: Type[nn.Module]) -> None:
        if not inspect.isclass(cls) or not issubclass(cls, nn.Module):
            raise TypeError(f"{cls!r} is not an nn.Module subclass")
        self._store[name] = cls

    def register_many(self, items: Mapping[str, Type[nn.Module]]) -> None:
        for name, cls in items.items():
            self.register(name, cls)

    def register_from_file(
        self,
        registration: FileRegistration,
    ) -> None:
        module = self._import_module_from_path(registration.file)
        exported = self._collect_nn_modules(module)
        if registration.class_name is None:
            selected = exported
        else:
            try:
                cls = exported[registration.class_name]
            except KeyError as exc:  # pragma: no cover - defensive branch
                available = ", ".join(sorted(exported)) or "<none>"
                raise KeyError(
                    f"Class {registration.class_name!r} not found in"
                    f" {registration.file}. Available: {available}"
                ) from exc
            selected = {registration.alias or registration.class_name: cls}
        self.register_many(selected)

    def register_from_files(self, registrations: Sequence[FileRegistration]) -> None:
        for registration in registrations:
            self.register_from_file(registration)

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------
    def resolve(self, target: str) -> Type[nn.Module]:
        if target in self._store:
            return self._store[target]
        module_path, _, attr = target.rpartition(".")
        if not module_path:
            raise KeyError(
                f"Unknown module {target!r}. Either register it explicitly or"
                " use a fully qualified import path."
            )
        module = importlib.import_module(module_path)
        resolved = getattr(module, attr)
        if not inspect.isclass(resolved) or not issubclass(resolved, nn.Module):
            raise TypeError(f"Resolved object {resolved!r} is not an nn.Module subclass")
        return resolved

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def _collect_nn_modules(self, module: ModuleType) -> Dict[str, Type[nn.Module]]:
        exported: Dict[str, Type[nn.Module]] = {}
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, nn.Module) and obj is not nn.Module:
                exported[name] = obj
        return exported

    def _import_module_from_path(self, path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"Unable to create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[call-arg]
        return module

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_files(cls, registrations: Iterable[Mapping[str, str]]) -> "ModuleRegistry":
        """Factory that builds a registry directly from dictionaries.

        Each mapping needs to provide at least a ``file`` key. Optional keys:
        ``class_name`` to narrow down the exported class and ``alias`` to give it
        a custom name inside the registry.
        """

        file_regs = [
            FileRegistration(
                file=Path(entry["file"]).resolve(),
                class_name=entry.get("class_name"),
                alias=entry.get("alias"),
            )
            for entry in registrations
        ]
        registry = cls()
        registry.register_from_files(file_regs)
        return registry


__all__ = ["FileRegistration", "ModuleRegistry"]
