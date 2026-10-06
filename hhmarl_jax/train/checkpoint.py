"""Checkpoints and policy exports (owner: agent B).  docs/contract.md §10."""
from __future__ import annotations

from typing import Any


def save_checkpoint(run_dir: str, train_state: Any, meta: dict) -> str:
    """-> path {run_dir}/ckpt/iter_{iteration:07d}.msgpack (+ .json meta, ckpt/latest.json)."""
    raise NotImplementedError


def load_checkpoint(path: str, template: Any) -> tuple[Any, dict]:
    raise NotImplementedError


def save_policy(path: str, params: Any, meta: dict) -> None:
    """Frozen policy file: msgpack {"params": state_dict, "meta": json string}."""
    raise NotImplementedError


def load_policy(path: str) -> tuple[Any, dict]:
    """-> (params pytree rebuilt via init_* from meta["net_cfg"], meta)."""
    raise NotImplementedError
