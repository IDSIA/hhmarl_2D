"""Rewards (owner: agent A).  Spec §8 (lowlevel) and §7.1-7.2 (hier)."""
from __future__ import annotations

import jax

Array = jax.Array


def lowlevel_rewards(state_pre, state_post, tick_events, fire_penalty: Array, opp_stats: Array,
                     cfg) -> Array:
    """f32[N] reward for each agent slot (0 for slots not alive at action time)."""
    raise NotImplementedError


def hier_tick_rewards(state_post, tick_events, cfg) -> Array:
    """f32[N] per-tick commander reward (+1 kill, -1 killed, OOB penalty)."""
    raise NotImplementedError


def hier_action_assess(state, commander_actions: Array, key: jax.Array, params, cfg):
    """Commander-step pre-pass: situation rewards, invalid-target -0.1, opponent fight/escape and
    target sampling.  Returns (effective_action i32[A], reward f32[N])."""
    raise NotImplementedError
