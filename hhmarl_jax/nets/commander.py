"""Commander networks (owner: agent B).  Spec §10.5 (CommanderGru), docs/contract.md §6.3.

Params pytree (CommanderParams), one network for both aircraft types:
    {"shared": {"kernel": f32[500,500], "bias": f32[500]},
     "actor": {...}, "critic": {...}}
Archs: "gru" (code), "sa" (MHA over time, same chunking as fight SA), "fc" (obs->500->SHARED).
Hidden state per agent: h_act f32[H], h_val f32[H], H = CMD_HIDDEN (200); zeros for sa/fc.
"""
from __future__ import annotations

import jax

from hhmarl_jax.config import NetConfig
from hhmarl_jax.types import Params

Array = jax.Array


def init_commander(key: jax.Array, net_cfg: NetConfig, n_opp_hl: int = 2) -> Params:
    raise NotImplementedError


def commander_actor_apply(params: Params, net_cfg: NetConfig, obs: Array, h0: Array,
                          seq_mask: Array) -> tuple[Array, Array]:
    """obs f32[B, T, O], h0 f32[B, H] -> (logits f32[B, T, n_actions], h_T f32[B, H])."""
    raise NotImplementedError


def commander_critic_apply(params: Params, net_cfg: NetConfig, critic_in: Array, h0: Array,
                           seq_mask: Array) -> tuple[Array, Array]:
    """critic_in f32[B, T, 3*(O+1)] -> (V f32[B, T], h_T f32[B, H])."""
    raise NotImplementedError
