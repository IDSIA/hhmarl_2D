"""Fictitious self-play pools (owner: agent B).  Spec §5.2, docs/contract.md §8.3."""
from __future__ import annotations

from hhmarl_jax.types import OpponentPool


def build_opponent_pool(policies_dir: str, level: int, l5_opponents: str = "code",
                        pool_size: int = 1) -> OpponentPool:
    """L4: fight=[L3_fight]; L5 (code): fight=[L3_fight, L4_fight], escape=[L3_escape],
    probs uniform 1/3 each.  Escape L5: fight=[L5_fight]."""
    raise NotImplementedError
