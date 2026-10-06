"""Observation builders (owner: agent A).  Spec §6.  All outputs f32 in [0, 1]."""
from __future__ import annotations

import jax

from hhmarl_jax.config import EnvConfig

Array = jax.Array


def fight_obs(state, slot: Array, target: Array, friend: Array, cfg: EnvConfig,
              frozen_dist: Array | None = None) -> Array:
    """f32[26]; AC2 rows use [:24], tail zero.  Zeros if slot dead or target < 0."""
    raise NotImplementedError


def escape_obs(state, slot: Array, opp_order: Array, friend: Array, cfg: EnvConfig,
               frozen_dist: Array | None = None) -> Array:
    """f32[30]; AC2 rows use [:29].  Two nearest opponents (zero padded) + friend block."""
    raise NotImplementedError


def commander_obs(state, slot: Array, cfg: EnvConfig) -> Array:
    """f32[cmd_obs_dim(cfg.n_opp_hl)] for an agent slot; zeros if dead/no opponent."""
    raise NotImplementedError


def nearest_enemies(state, slot: Array) -> tuple[Array, Array]:
    """(enemy slots sorted by planar distance, stable; -1 padded, normalised distances)."""
    raise NotImplementedError
