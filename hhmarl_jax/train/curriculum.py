"""Curriculum driver (owner: agent B).  docs/contract.md §8.2."""
from __future__ import annotations

from hhmarl_jax.config import TrainConfig


def train_phase(cfg: TrainConfig, phase: str, restore_from: str | None = None) -> str:
    """Train one phase ("L{n}_fight" | "L{n}_escape"); returns its run dir."""
    raise NotImplementedError


def run_curriculum(cfg: TrainConfig) -> dict[str, str]:
    """Run cfg.curriculum.phases in order (commander phase delegates to train.commander)."""
    raise NotImplementedError
