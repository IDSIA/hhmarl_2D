"""Sequential simulator tick (owner: agent A).  Spec §2.3-2.5.

Randomness is drawn UP FRONT into explicit pytrees so parity tests can inject the exact draws the
original code consumed (docs/contract.md §12).
"""
from __future__ import annotations

from typing import NamedTuple

import jax

from hhmarl_jax.config import EnvConfig
from hhmarl_jax.env.state import AircraftState, RocketState

Array = jax.Array


class TickDraws(NamedTuple):
    cannon_u: Array       # f32[A, A] U(0,1) for (shooter, target); hit if u < hit_prob/burst_s
    rocket_noise: Array   # f32[A]    U(0.95, 1.05) heading noise per rocket-owner slot


class TickEvents(NamedTuple):
    """Kills emitted during one tick, in emission order semantics flattened to matrices."""
    killed: Array         # bool[A]   aircraft destroyed this tick (cannon or rocket)
    killer: Array         # i32[A]    killer aircraft slot (-1 none); rocket kills -> source slot
    by_rocket: Array      # bool[A]
    oob: Array            # bool[A]   removed out of bounds after the tick


def sample_tick_draws(key: jax.Array, cfg: EnvConfig) -> TickDraws:
    raise NotImplementedError


def tick(aircraft: AircraftState, rockets: RocketState, t: Array, draws: TickDraws,
         cfg: EnvConfig) -> tuple[AircraftState, RocketState, TickEvents]:
    """One 1-s tick: aircraft slots in order (heading, speed, cannon, missile steer, move), then
    rockets in launch_id order (hit check, slot-1 fuse quirk, lifetime, turn, speed, move), then
    OOB removal.  Units dead at tick start do not update; units killed during the tick still do
    (quirk).  Pure; jit/vmap safe."""
    raise NotImplementedError


def try_fire_missile(aircraft: AircraftState, rockets: RocketState, shooter: Array,
                     target: Array, t: Array, next_launch_id: Array,
                     cfg: EnvConfig) -> tuple[AircraftState, RocketState, Array, Array]:
    """fire_missile with offset radar cone; returns (aircraft, rockets, launched: bool, next_id)."""
    raise NotImplementedError
