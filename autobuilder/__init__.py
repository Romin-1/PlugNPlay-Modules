"""Utility package for dynamically assembling and training models from the
PlugNPlay module zoo.

The submodules expose:

- :mod:`autobuilder.registry` for loading PyTorch ``nn.Module`` classes from the
  repository or standard libraries.
- :mod:`autobuilder.graph` for turning a declarative configuration into an
  executable computation graph.
- :mod:`autobuilder.trainer` for a lightweight training loop that works with the
  dynamically assembled models.

These helpers are intentionally simple so they can be copy/pasted into a
research project and then customised as required.
"""

from .graph import build_model_from_config  # noqa: F401
from .registry import ModuleRegistry  # noqa: F401
from .trainer import Trainer, TrainerConfig  # noqa: F401
