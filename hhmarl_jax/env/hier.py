"""Hierarchical commander env (owner: agent A).  docs/contract.md §4.4.

One step = one commander decision = 1..16 sim ticks, run with lax.scan over hl_max_substeps
ticks plus an `active` mask (no data-dependent Python loops).  Frozen low-level policies for
both teams come in through HierEnvParams (params) + logits fns (static, constructor).
"""
from __future__ import annotations

from typing import Tuple

import jax

from hhmarl_jax.config import EnvConfig
from hhmarl_jax.env.state import HierState
from hhmarl_jax.types import HierEnvParams, HierInfo, LogitsFn, LowLevelPolicySet

Array = jax.Array


class HierEnv:
    def __init__(self, cfg: EnvConfig, ll_logits_fns: Tuple[LogitsFn, LogitsFn]):
        """ll_logits_fns = (fight_fn, escape_fn); the same functions are used for both teams,
        with params from HierEnvParams.ll_agent / ll_opp."""
        self.cfg = cfg
        self.ll_logits_fns = ll_logits_fns

    @property
    def obs_dim(self) -> int:
        """cmd_obs_dim(cfg.n_opp_hl) (34 by default)."""
        raise NotImplementedError

    @property
    def num_actions(self) -> int:
        """1 + cfg.n_opp_hl (3 by default)."""
        raise NotImplementedError

    def default_params(self, ll_agent: LowLevelPolicySet,
                       ll_opp: LowLevelPolicySet | None = None) -> HierEnvParams:
        """num_* = *_max, ll_opp = ll_agent if None, opp_fight_ratio = 0.75."""
        raise NotImplementedError

    def reset(self, key: jax.Array, params: HierEnvParams) -> tuple[Array, HierState]:
        """-> (obs f32[N, obs_dim], state)."""
        raise NotImplementedError

    def step_env(self, key: jax.Array, state: HierState, actions: Array, params: HierEnvParams):
        """actions i32[N] in [0, num_actions).  No auto-reset.
        -> (obs f32[N,obs_dim], state, reward f32[N], done bool[], info HierInfo)."""
        raise NotImplementedError

    def step(self, key: jax.Array, state: HierState, actions: Array,
             params: HierEnvParams) -> tuple[Array, HierState, Array, Array, HierInfo]:
        """step_env + auto-reset (same semantics as LowLevelEnv.step)."""
        raise NotImplementedError
