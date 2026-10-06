"""Scripted opponents L1/L2/L3 (owner: agent A).  Spec §5.1."""
from __future__ import annotations

from typing import NamedTuple

import jax

from hhmarl_jax.config import EnvConfig

Array = jax.Array


class ScriptDraws(NamedTuple):
    """All random draws scripted opponents may consume in one step (fields are agent A's choice;
    must be documented so tests/ can inject them)."""
    u: Array              # f32[M, k] uniforms, k fixed by agent A


def sample_script_draws(key: jax.Array, cfg: EnvConfig) -> ScriptDraws:
    raise NotImplementedError


def scripted_opponent_step(state, draws: ScriptDraws, cfg: EnvConfig):
    """Set new_heading/new_speed/burst and attempt missiles for opponent slots at cfg.level in
    {1,2,3}.  Returns updated LowLevelState (pre-tick)."""
    raise NotImplementedError
