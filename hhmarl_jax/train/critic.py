"""Centralized-critic inputs (owner: agent B).  docs/contract.md §7.2."""
from __future__ import annotations

import jax

Array = jax.Array


def scale_lowlevel_actions(actions: Array, ac_type: Array) -> Array:
    """i32[..., 4] -> f32[..., 4]: (a0/12, a1/8, a2, a3); a3 dropped (0) for AC2."""
    raise NotImplementedError


def lowlevel_critic_input(obs: Array, actions: Array | None, ac_type: Array) -> Array:
    """2-agent team.  obs f32[..., 2, D], actions i32[..., 2, 4] or None (zeros), ac_type i32[..., 2]
    -> f32[..., 2, C] with per-row layout [act_own | act_other | obs_own | obs_other] using the
    UNPADDED per-type widths (AC1 row: 4|3|26|24, AC2 row: 3|4|24|26 for fight)."""
    raise NotImplementedError


def commander_critic_input(obs: Array, actions: Array | None, n_actions: int) -> Array:
    """3-agent team.  obs f32[..., 3, O], actions i32[..., 3] or None -> f32[..., 3, 3*(O+1)]
    layout [a_own, a_lo, a_hi, obs_own, obs_lo, obs_hi], a scaled by 1/(n_actions-1); lo/hi =
    the other two agents in slot order."""
    raise NotImplementedError
