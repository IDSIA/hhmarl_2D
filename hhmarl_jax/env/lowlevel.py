"""Low-level combat env (owner: agent A).  docs/contract.md §4.3.

Usage (trainer):
    env = LowLevelEnv(EnvConfig.lowlevel(level=3), opp_logits_fns=(fight_fn, escape_fn))
    params = env.default_params(opp_pool)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
    obs, state, rew, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(keys, state, act, params)
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax

from hhmarl_jax.config import EnvConfig
from hhmarl_jax.env.state import LowLevelState
from hhmarl_jax.types import LogitsFn, LowLevelEnvParams, LowLevelInfo, OpponentPool

Array = jax.Array


class LowLevelEnv:
    """Pure-functional env.  The instance only holds STATIC things (cfg, policy functions);
    all methods are jit/vmap-safe pure functions of their array arguments."""

    def __init__(self, cfg: EnvConfig,
                 opp_logits_fns: Optional[Tuple[LogitsFn, LogitsFn]] = None):
        """opp_logits_fns = (fight_fn, escape_fn), required iff cfg.level >= 4.
        Raises NotImplementedError for unsupported quirk/paper switches."""
        self.cfg = cfg
        self.opp_logits_fns = opp_logits_fns

    # ------------------------------------------------------------------ dims
    @property
    def obs_dim(self) -> int:
        """26 (fight) or 30 (escape): padded obs width D."""
        raise NotImplementedError

    def default_params(self, opp_pool: Optional[OpponentPool] = None) -> LowLevelEnvParams:
        """num_agents=n_agents_max, num_opps=n_opps_max; a dummy pool (zeros) if None."""
        raise NotImplementedError

    # ------------------------------------------------------------------ API
    def reset(self, key: jax.Array, params: LowLevelEnvParams) -> tuple[Array, LowLevelState]:
        """-> (obs f32[N, D], state)."""
        raise NotImplementedError

    def step_env(self, key: jax.Array, state: LowLevelState, actions: Array,
                 params: LowLevelEnvParams):
        """One env step WITHOUT auto-reset.  actions i32[N, 4].
        -> (obs f32[N,D], state, reward f32[N], done bool[], info LowLevelInfo)."""
        raise NotImplementedError

    def step(self, key: jax.Array, state: LowLevelState, actions: Array,
             params: LowLevelEnvParams) -> tuple[Array, LowLevelState, Array, Array, LowLevelInfo]:
        """step_env + auto-reset: if done, returned obs/state belong to a fresh episode
        (reset with a key split from `key`); info.final_obs / info.episode describe the finished
        one."""
        raise NotImplementedError
