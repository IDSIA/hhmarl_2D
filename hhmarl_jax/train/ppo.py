"""PPO (owner: agent B).  Spec §10.2, docs/contract.md §8."""
from __future__ import annotations

from typing import NamedTuple

import jax

from hhmarl_jax.config import PPOConfig, RLlibQuirks

Array = jax.Array


class TrainState(NamedTuple):
    params: object        # LowLevelParams or CommanderParams
    opt_state: object     # dict policy_name -> optax state ("ac1","ac2" | "commander" | "joint")
    kl_coeff: object      # dict policy_name -> f32[]
    iteration: Array      # i32[]
    env_steps: Array      # i32[] cumulative env steps of this phase (int64 not needed < 2^31)
    key: jax.Array


def compute_gae(reward: Array, value: Array, done: Array, truncated: Array, final_value: Array,
                last_value: Array, gamma: float, lam: float, bootstrap_truncation: bool):
    """Time-major [T, B, N].  -> (advantages, value_targets).  Faithful: truncation == terminal."""
    raise NotImplementedError


def make_update_fn(net_cfg, ppo_cfg: PPOConfig, quirks: RLlibQuirks, kind: str):
    """-> update(train_state, traj, last_value) -> (train_state, metrics: dict[str, f32[]])."""
    raise NotImplementedError
