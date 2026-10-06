"""Env state pytrees (owner: agent A).

PINNED fields (docs/contract.md §4.2) may not be renamed/removed; agent A may APPEND fields.
All arrays are per env (no batch axis); vmap adds the leading batch axis.
Slot order == original unit-id order: slots 0..N-1 agents (ids 1..N), N..A-1 opponents.
"""
from __future__ import annotations

from typing import NamedTuple

import jax

Array = jax.Array


class AircraftState(NamedTuple):
    """Arrays of shape [A]."""
    lat: Array            # f32 degrees, offset from MAP_ORIGIN_LAT (lat - 5.0)
    lon: Array            # f32 degrees, offset from MAP_ORIGIN_LON (lon - 7.0)
    heading: Array        # f32 deg, compass, [0, 360)
    speed: Array          # f32 knots
    new_heading: Array    # f32 commanded heading (set before the tick)
    new_speed: Array      # f32 commanded speed
    alive: Array          # bool, in the sim (not destroyed / not OOB)
    exists: Array         # bool, slot used in this episode
    is_agent: Array       # bool, True for slots < n_agents_max
    ac_type: Array        # i32, 1 or 2
    cannon_remain: Array  # f32 seconds of burst left
    cannon_max: Array     # f32 normaliser (after per-level override)
    burst: Array          # f32 seconds of current burst left (> 0 == shooting)
    missile_remain: Array  # i32
    rocket_max: Array     # i32 normaliser (0 for AC2)
    missile_wait: Array   # i32 cooldown
    has_missile: Array    # bool, original `actual_missile is not None`


class RocketState(NamedTuple):
    """Arrays of shape [A]; slot k holds the (single) rocket launched by aircraft slot k."""
    active: Array         # bool
    lat: Array            # f32 offset degrees
    lon: Array            # f32
    heading: Array        # f32 (may leave [0,360) transiently, quirk)
    new_heading: Array    # f32
    speed: Array          # f32 knots
    fire_tick: Array      # i32 sim tick at launch (life = t - fire_tick)
    target: Array         # i32 aircraft slot
    launch_id: Array      # i32 monotonically increasing; rockets update in launch_id order


class LowLevelState(NamedTuple):
    aircraft: AircraftState
    rockets: RocketState
    t: Array              # i32 env steps == sim ticks taken in this episode
    side: Array           # i32 r in {1, 2} (spec §4.2)
    opp_to_attack: Array  # i32[A] current target slot of each aircraft, -1 = none
    opp_pool_idx: Array   # i32 FSP pool entry for this episode (-1 scripted)
    opp_mode: Array       # i32 MODE_FIGHT / MODE_ESCAPE for FSP opponents
    l3_escaping: Array    # bool shared scripted-L3 escape flag
    l3_escape_timer: Array  # i32
    next_launch_id: Array  # i32
    num_agents: Array     # i32 copied from params at reset
    num_opps: Array       # i32
    episode: "object"     # hhmarl_jax.types.EpisodeStats running accumulators
    # ---- agent A may append private fields below this line ----


class HierState(NamedTuple):
    aircraft: AircraftState
    rockets: RocketState
    t: Array              # i32 sim ticks taken in this episode (may exceed 500, up to 515)
    cmd_t: Array          # i32 commander steps taken
    side: Array           # i32
    opp_sorted: Array     # i32[A, max(N,M)] per aircraft: enemy slots sorted by planar distance
                          #   at the start of the commander step (-1 padded)  [frozen list]
    opp_sorted_dist: Array  # f32[A, max(N,M)] normalised distances, frozen (quirk)
    cmd_action: Array     # i32[A] effective commander action (-1 dead)
    next_launch_id: Array
    num_agents: Array
    num_opps: Array
    episode: "object"     # EpisodeStats
    # ---- agent A may append private fields below this line ----
