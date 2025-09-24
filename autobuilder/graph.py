"""Declarative graph builder for PlugNPlay modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn

from .registry import ModuleRegistry


@dataclass
class NodeConfig:
    name: str
    target: str | None = None
    op: str | None = None
    init: Dict[str, Any] | None = None
    call: Dict[str, Any] | None = None
    inputs: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if (self.target is None) == (self.op is None):
            raise ValueError(
                f"Node {self.name!r} must specify exactly one of 'target' or 'op'"
            )
        if not self.inputs:
            self.inputs = []
        else:
            self.inputs = list(self.inputs)
        if self.call is None:
            self.call = {}
        if self.init is None:
            self.init = {}


class DynamicGraphModel(nn.Module):
    """Small utility that executes the declarative computation graph."""

    def __init__(
        self,
        modules: Dict[str, nn.Module],
        graph: Sequence[NodeConfig],
        input_names: Sequence[str],
        outputs: Sequence[str],
    ) -> None:
        super().__init__()
        self.outputs = list(outputs)
        self.input_names = list(input_names)
        self.modules_map = nn.ModuleDict(modules)
        self.graph = list(graph)

    def forward(self, *args: torch.Tensor, **named_inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        cache: Dict[str, torch.Tensor]
        if args:
            if named_inputs:
                raise ValueError("Pass either positional or keyword inputs, not both")
            if len(args) != len(self.input_names):
                raise ValueError(
                    f"Expected {len(self.input_names)} positional inputs but got {len(args)}"
                )
            cache = dict(zip(self.input_names, args))
        else:
            missing = [name for name in self.input_names if name not in named_inputs]
            if missing:
                raise ValueError(f"Missing required inputs: {missing}")
            cache = {name: named_inputs[name] for name in self.input_names}
        for node in self.graph:
            args = [cache[name] for name in node.inputs]
            if node.target is not None:
                module = self.modules_map[node.name]
                if args:
                    result = module(*args, **node.call)
                elif node.call:
                    result = module(**node.call)
                else:
                    result = module()
            else:
                result = self._execute_op(node.op, args, node.call)
            cache[node.name] = result
        collected = tuple(cache[name] for name in self.outputs)
        return collected[0] if len(collected) == 1 else collected

    # pylint: disable=too-many-return-statements
    def _execute_op(self, op: str | None, args: Sequence[torch.Tensor], params: Dict[str, Any]) -> torch.Tensor:
        if op == "add":
            if not args:
                raise ValueError("add op requires at least one input tensor")
            result = args[0]
            for tensor in args[1:]:
                result = result + tensor
            return result
        if op == "cat":
            dim = params.get("dim", 1)
            if not args:
                raise ValueError("cat op requires input tensors")
            return torch.cat(args, dim=dim)
        if op == "mul":
            if not args:
                raise ValueError("mul op requires at least one input tensor")
            result = args[0]
            for tensor in args[1:]:
                result = result * tensor
            return result
        if op == "identity":
            return args[0]
        if op == "flatten":
            start_dim = params.get("start_dim", 1)
            end_dim = params.get("end_dim", -1)
            return torch.flatten(args[0], start_dim=start_dim, end_dim=end_dim)
        raise ValueError(f"Unsupported op {op!r}")


def build_model_from_config(config: Mapping[str, Any], registry: ModuleRegistry) -> DynamicGraphModel:
    model_cfg = config["model"]
    nodes_cfg = [NodeConfig(**node) for node in model_cfg["nodes"]]
    modules: Dict[str, nn.Module] = {}
    for node in nodes_cfg:
        if node.target is None:
            continue
        cls = registry.resolve(node.target)
        modules[node.name] = cls(**node.init)
    input_names = model_cfg.get("inputs", [])
    if not input_names:
        raise ValueError("Model config must declare at least one input name")
    outputs = model_cfg.get("outputs", [nodes_cfg[-1].name])
    return DynamicGraphModel(modules, nodes_cfg, input_names, outputs)
