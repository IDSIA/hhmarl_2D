"""Episode recording for renders/animations (owner: agent D).  docs/contract.md §11.3."""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class EpisodeRecord(NamedTuple):
    """numpy arrays at SIM-TICK resolution; F = number of ticks recorded."""
    lat: np.ndarray            # f32[F, A] absolute degrees
    lon: np.ndarray            # f32[F, A]
    heading: np.ndarray        # f32[F, A]
    alive: np.ndarray          # bool[F, A]
    shooting: np.ndarray       # bool[F, A]
    rocket_active: np.ndarray  # bool[F, A]
    rocket_lat: np.ndarray     # f32[F, A]
    rocket_lon: np.ndarray     # f32[F, A]
    is_agent: np.ndarray       # bool[A]
    ac_type: np.ndarray        # i32[A]
    cmd_action: np.ndarray     # i32[F, A] effective commander action per tick (-1 / lowlevel)
    reward: np.ndarray         # f32[F, N]
    meta: dict                 # map_size, kind, levels, outcome ("win"|"loss"|"draw"|"none")


def record_episode(key: Any, cfg, agent_ll, opp_ll, commander_params=None, **kw) -> EpisodeRecord:
    raise NotImplementedError


def save_record(rec: EpisodeRecord, path: str) -> None:
    """np.savez_compressed; meta stored as a JSON string under key "meta"."""
    raise NotImplementedError


def load_record(path: str) -> EpisodeRecord:
    raise NotImplementedError
