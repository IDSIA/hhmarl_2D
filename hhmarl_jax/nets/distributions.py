"""Action distributions (owner: agent B).  docs/contract.md §6.4.

Low-level: MultiCategorical over ACTION_NVEC = (13, 9, 2, 2) on logits f32[..., 26].
For ac_type == 2 the 4th component (logits[24:26]) is IGNORED: sampled/mode action a[3] = 0 and
its logp / entropy / KL contributions are 0.  Commander: Categorical(n_actions).
"""
from __future__ import annotations

import jax

Array = jax.Array


def pad_ac2_logits(logits24: Array) -> Array:
    """f32[..., 24] -> f32[..., 26] (zeros appended)."""
    raise NotImplementedError


def multicat_sample(key: jax.Array, logits: Array, ac_type: Array) -> Array:
    """-> i32[..., 4]."""
    raise NotImplementedError


def multicat_mode(logits: Array, ac_type: Array) -> Array:
    """Per-component argmax (FSP / deterministic eval) -> i32[..., 4]."""
    raise NotImplementedError


def multicat_logp(logits: Array, actions: Array, ac_type: Array) -> Array:
    """Sum of component log-probs -> f32[...]."""
    raise NotImplementedError


def multicat_entropy(logits: Array, ac_type: Array) -> Array:
    raise NotImplementedError


def multicat_kl(logits_old: Array, logits_new: Array, ac_type: Array) -> Array:
    """KL(old || new), summed over components."""
    raise NotImplementedError


def categorical_sample(key: jax.Array, logits: Array) -> Array:
    raise NotImplementedError


def categorical_logp(logits: Array, actions: Array) -> Array:
    raise NotImplementedError


def categorical_entropy(logits: Array) -> Array:
    raise NotImplementedError


def categorical_kl(logits_old: Array, logits_new: Array) -> Array:
    raise NotImplementedError
