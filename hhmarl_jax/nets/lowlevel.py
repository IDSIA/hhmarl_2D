"""Low-level actor-critic networks (owner: agent B).  Spec §10.5, docs/contract.md §6.1-6.2.

Params pytree (LowLevelParams), one tree per (mode, experiment):
    {"shared": {"kernel": f32[500,500], "bias": f32[500]},      # SHARED_LAYER
     "ac1": {"actor": {...}, "critic": {...}},
     "ac2": {"actor": {...}, "critic": {...}}}
Inner leaf names are agent B's choice but must be stable once checkpoints exist.
Architectures:
  fight "sa":  actor  own->200, rest->200, full->100 + MHA(100,2) + skip + norm, concat 500
               -> SHARED -> linear 26/24;
               critic [own obs+act]->175, [other obs+act]->175, all->150 + MHA(150,2) + skip
               + norm -> SHARED -> linear 1.
  fight "fc":  actor obs->500 tanh -> SHARED -> logits; critic C->500 tanh -> SHARED -> 1.
  escape:      actor own->150, opps->250, friend->100 -> SHARED -> logits;
               critic C(66)->500 -> SHARED -> 1.
"""
from __future__ import annotations

from typing import Literal

import jax

from hhmarl_jax.config import NetConfig
from hhmarl_jax.types import LogitsFn, Params

Array = jax.Array
Mode = Literal["fight", "escape"]


def init_lowlevel(key: jax.Array, net_cfg: NetConfig, mode: Mode) -> Params:
    """-> LowLevelParams (layout above)."""
    raise NotImplementedError


def actor_apply(params: Params, net_cfg: NetConfig, mode: Mode, ac_type: int, obs: Array,
                seq_mask: Array) -> Array:
    """ac_type is a static python int (1|2).
    obs f32[B, T, OBS_*[ac_type]] (UNPADDED width), seq_mask bool[B, T] (True = real step).
    -> logits f32[B, T, LOGITS_DIM_AC[ac_type]].  With sa_mode="rllib" attention is non-causal over
    the whole T axis and ignores seq_mask (quirk); T=1 at rollout."""
    raise NotImplementedError


def critic_apply(params: Params, net_cfg: NetConfig, mode: Mode, ac_type: int,
                 critic_in: Array, seq_mask: Array) -> Array:
    """critic_in f32[B, T, C] (C = 57 fight / 66 escape, layout contract §7.2) -> V f32[B, T]."""
    raise NotImplementedError


def policy_logits_padded(params: Params, net_cfg: NetConfig, mode: Mode, obs: Array,
                         ac_type: Array) -> Array:
    """Rollout helper, T=1, both type heads evaluated and selected.
    obs f32[..., D] (padded), ac_type i32[...] -> logits f32[..., 26] (AC2 zero-padded)."""
    raise NotImplementedError


def value_padded(params: Params, net_cfg: NetConfig, mode: Mode, critic_in: Array,
                 ac_type: Array) -> Array:
    """Rollout helper, T=1: critic_in f32[..., C], ac_type i32[...] -> f32[...]."""
    raise NotImplementedError


def make_logits_fn(net_cfg: NetConfig, mode: Mode) -> LogitsFn:
    """-> fn(params, obs f32[D], ac_type i32[]) -> f32[26]; the function handed to the envs."""
    raise NotImplementedError
