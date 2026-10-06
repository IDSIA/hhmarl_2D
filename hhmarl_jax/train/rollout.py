"""Vectorized rollouts (owner: agent B).  docs/contract.md §7.1."""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax

from hhmarl_jax.types import Transition

Array = jax.Array


class RunnerState(NamedTuple):
    env_state: object     # batched LowLevelState / HierState
    obs: Array            # f32[B, N, D]
    ep_step: Array        # i32[B]
    h_act: Array          # f32[B, N, H] (H=0 for non-recurrent)
    h_val: Array
    key: jax.Array


def make_rollout(env, net_cfg, quirks, num_envs: int, rollout_len: int,
                 kind: str = "lowlevel") -> tuple[Callable, Callable]:
    """-> (init_fn(key, env_params) -> RunnerState,
           rollout_fn(params, runner, env_params) -> (runner, Transition[T=rollout_len, B], last_value f32[B,N])).
    Both jit-able; env is vmapped over B with env_params in_axes=None."""
    raise NotImplementedError
