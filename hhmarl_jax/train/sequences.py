"""RLlib-style sequence chunking (owner: agent B).  Spec §10.5, docs/contract.md §7.3."""
from __future__ import annotations

from typing import NamedTuple

import jax

Array = jax.Array


class SeqBatch(NamedTuple):
    """Rows = sequences of length L = max_seq_len, zero padded."""
    data: object          # pytree of per-step fields with leading [S, L]
    mask: Array           # bool[S, L] True = real step
    h_act0: Array         # f32[S, H] recurrent state at chunk start
    h_val0: Array


def chunk(traj, max_seq_len: int) -> SeqBatch:
    """Split each (env, agent) stream at episode boundaries, rollout boundaries and every
    max_seq_len steps of the episode (aligned to ep_step % max_seq_len == 0, as RLlib)."""
    raise NotImplementedError
