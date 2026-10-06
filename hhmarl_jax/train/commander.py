"""Commander training (owner: agent B).  docs/contract.md §8.4."""
from __future__ import annotations

from hhmarl_jax.config import TrainConfig


def train_commander(cfg: TrainConfig, policies_dir: str) -> str:
    """Frozen LL policies: L5_fight + (L5_escape if present else L3_escape) for both teams.
    3v3 HierEnv, PPOConfig.commander().  Returns run dir."""
    raise NotImplementedError
