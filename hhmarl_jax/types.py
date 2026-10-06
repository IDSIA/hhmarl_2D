"""Public pytrees exchanged between agents.  CONTRACT-FROZEN (owner: orchestrator).

Shapes use: B = batch of envs, T = time, N = n_agents_max, M = n_opps_max, A = N + M slots,
D = padded low-level obs dim (26 fight / 30 escape), C = critic input dim, K = pool size,
S = hl_max_substeps (16).  "f32" = jnp.float32, "i32" = jnp.int32.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax

Array = jax.Array
Params = Any        # a Flax params pytree (nested dict of arrays)
PRNGKey = jax.Array

# ======================================================================== policy plumbing

LogitsFn = Callable[[Params, Array, Array], Array]
"""Frozen low-level policy forward used INSIDE the env (spec §5.2):
    fn(params, obs: f32[D], ac_type: i32[]) -> logits: f32[26]
Single-sample, T=1 (no recurrence).  The env vmaps it over aircraft slots.  Must handle both
aircraft types (compute both type heads and select with ac_type; AC2 logits padded 24->26 with 0).
Built by hhmarl_jax.nets.lowlevel.make_logits_fn(net_cfg, mode)."""


class LowLevelPolicySet(NamedTuple):
    """Frozen low-level params used to drive aircraft inside an env (deterministic argmax)."""
    fight: Params      # LowLevelParams of a fight model (nets/lowlevel.py layout)
    escape: Params     # LowLevelParams of an escape model


class OpponentPool(NamedTuple):
    """FSP opponents for lowlevel L4/L5 (spec §5.2).  Leaves of fight/escape have a leading
    pool axis.  Env samples one entry per episode at reset: idx ~ Categorical(probs), where
    idx < K_F selects fight[idx] (opp_mode fight) and idx >= K_F selects escape[idx-K_F].
    Keep K_F + K_E small (<= 4): the env evaluates every entry and selects (no param gather)."""
    fight: Params      # leading axis K_F (>= 1)
    escape: Params     # leading axis K_E (>= 1; a dummy copy with prob 0 if unused)
    probs: Array       # f32[K_F + K_E], sums to 1


class LowLevelEnvParams(NamedTuple):
    """Dynamic lowlevel env params.  Passed UNBATCHED (vmap in_axes=None) in training."""
    num_agents: Array            # i32[] <= n_agents_max
    num_opps: Array              # i32[] <= n_opps_max
    opp_pool: OpponentPool       # used for level >= 4 (dummy otherwise)


class HierEnvParams(NamedTuple):
    num_agents: Array            # i32[]
    num_opps: Array              # i32[]
    ll_agent: LowLevelPolicySet  # drives agents (code: L5 fight + L5/L3 escape)
    ll_opp: LowLevelPolicySet    # drives opponents (training: same as ll_agent; eval fight: L4)
    opp_fight_ratio: Array       # f32[] P(opponent fights) per commander step (0.75)


# ======================================================================== env outputs


class StepEvents(NamedTuple):
    """Event counters for ONE env step (lowlevel tick or commander step).  Index [t-1] = AC type t."""
    kills: Array            # i32[2]  opponents destroyed by agents, by KILLER type (k-1, k-2)
    killed_by_opp: Array    # i32[2]  agents destroyed by opponents, by VICTIM type
    oob: Array              # i32[2]  agents removed out of bounds, by type
    fk_by_killer: Array     # i32[2]  friendly kills (agent killed agent), by killer type
    fk_by_victim: Array     # i32[2]  friendly kills, by victim type
    opp_lost_other: Array   # i32[]   opponents lost to OOB or opponent fire
    rocket_kills: Array     # i32[]   opponents destroyed by agent rockets (subset of kills)


class EpisodeStats(NamedTuple):
    """Accumulated over the episode; read it from info.episode where info.done is True."""
    ret: Array              # f32[]   team return = sum over agents of undiscounted rewards
    ret_per_agent: Array    # f32[N]
    length: Array           # i32[]   env steps (lowlevel ticks / commander steps)
    ticks: Array            # i32[]   sim ticks (== length for lowlevel)
    win: Array              # bool[]  alive_opps == 0 and t < horizon
    loss: Array             # bool[]  alive_agents == 0 and t < horizon
    draw: Array             # bool[]  t >= horizon and both teams alive
    events: StepEvents      # summed over the episode
    cmd_counts: Array       # i32[6]  agent_fight, agent_escape, opp_fight, opp_escape, opp1, opp2
                            #         (hier only; zeros in lowlevel)


class LowLevelInfo(NamedTuple):
    terminated: Array       # bool[]  a team was wiped out
    truncated: Array        # bool[]  t >= horizon and not terminated
    done: Array             # bool[]  terminated | truncated  (episode-level; no per-agent dones)
    alive_at_action: Array  # bool[N] agent alive at the start of the step (RLlib reward keys)
    alive: Array            # bool[N] agent alive after the step
    exists: Array           # bool[N] slot used this episode (i < num_agents)
    ac_type: Array          # i32[N]  1 or 2 (of the episode that produced this step)
    final_obs: Array        # f32[N,D] obs after this step BEFORE auto-reset (== obs if not done)
    events: StepEvents
    episode: EpisodeStats   # running stats of the episode that produced this step


class TickRecord(NamedTuple):
    """Per-sub-tick snapshot for rendering (hier env with EnvConfig.record_ticks=True)."""
    valid: Array            # bool[S] sub-tick actually simulated
    lat: Array              # f32[S,A] degrees offset from MAP_ORIGIN_LAT
    lon: Array              # f32[S,A]
    heading: Array          # f32[S,A]
    alive: Array            # bool[S,A]
    shooting: Array         # bool[S,A] cannon burst > 0
    rocket_active: Array    # bool[S,A] rocket slot k belongs to shooter slot k
    rocket_lat: Array       # f32[S,A]
    rocket_lon: Array       # f32[S,A]


class HierInfo(NamedTuple):
    terminated: Array
    truncated: Array
    done: Array
    alive_at_action: Array  # bool[N]
    alive: Array            # bool[N]
    exists: Array           # bool[N]
    ac_type: Array          # i32[N]
    final_obs: Array        # f32[N, cmd_obs_dim]
    n_ticks: Array          # i32[] sim ticks consumed by this commander step (1..16)
    effective_action: Array  # i32[A] post-assessment commander action of every slot (-1 dead)
    events: StepEvents
    episode: EpisodeStats
    ticks: Any              # TickRecord if cfg.record_ticks else None


# ======================================================================== rollout -> PPO


class Transition(NamedTuple):
    """Time-major rollout batch: leading axes [T, B], agent axis N.  Built by train/rollout.py.

    Lowlevel: obs D = 26/30, action i32[...,4], logits f32[...,26].
    Commander: obs D = cmd_obs_dim, action i32[...] (no trailing dim), logits f32[...,n_actions],
               h_act/h_val = recurrent state fed INTO this step (zeros for non-GRU archs).
    """
    obs: Array              # f32[T,B,N,D]
    critic_obs: Array       # f32[T,B,N,C]  zero action slots (as seen at rollout)
    action: Array           # i32[T,B,N,4] | i32[T,B,N]
    logits: Array           # f32[T,B,N,L]  behaviour logits (RLlib ACTION_DIST_INPUTS)
    logp: Array             # f32[T,B,N]
    value: Array            # f32[T,B,N]    V at rollout (zero-action critic input if faithful)
    reward: Array           # f32[T,B,N]
    done: Array             # bool[T,B]
    terminated: Array       # bool[T,B]
    truncated: Array        # bool[T,B]
    alive: Array            # bool[T,B,N]   alive_at_action
    exists: Array           # bool[T,B,N]
    ac_type: Array          # i32[T,B,N]
    ep_step: Array          # i32[T,B]      step index within the episode (0-based), chunk alignment
    final_value: Array      # f32[T,B,N]    V(final_obs) (used only for clean truncation bootstrap)
    h_act: Array            # f32[T,B,N,H]  commander only (H=200); shape [T,B,N,0] otherwise
    h_val: Array            # f32[T,B,N,H]


# ======================================================================== evaluation


class EvalResult(NamedTuple):
    """Totals over n_episodes (python ints/floats after device_get).  See contract §11."""
    n_episodes: int
    win: int
    loss: int
    draw: int
    kills: tuple            # (k-1, k-2)  by killer type
    killed_by_opp: tuple    # (d-1, d-2)  by victim type, shot down by opponents
    oob: tuple              # (oob-1, oob-2)
    fk_by_killer: tuple     # (fk-1, fk-2)
    fk_by_victim: tuple
    escaped: int            # Fig 8b categories (mutually exclusive, see contract §11)
    killed: int
    kills_cat: int
    mean_return: float
    mean_length: float
    cmd_counts: tuple       # agent_fight, agent_escape, opp_fight, opp_escape, opp1, opp2
    config: dict            # dataclasses.asdict(EvalConfig) + policy paths
