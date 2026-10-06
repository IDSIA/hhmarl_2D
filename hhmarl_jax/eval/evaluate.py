"""Batched evaluation returning win/loss/draw and per-type kill counts (owner: agent D)."""
from __future__ import annotations

from typing import Any, Optional

from hhmarl_jax.config import EvalConfig, NetConfig
from hhmarl_jax.types import EvalResult, LowLevelPolicySet


def evaluate(cfg: EvalConfig, agent_ll: LowLevelPolicySet, opp_ll: LowLevelPolicySet,
             commander_params: Optional[Any] = None, ll_net_cfg: NetConfig = NetConfig(),
             cmd_net_cfg: NetConfig = NetConfig()) -> EvalResult:
    """Run exactly cfg.n_episodes complete episodes (vectorized over cfg.num_envs envs; extra
    episodes finished in the last batch are discarded deterministically by env index).
    kind="commander": agents' commander = commander_params; "fight": every agent action 1
    (nearest opponent); "escape": every agent action 0."""
    raise NotImplementedError


def eval_sweep(cfg: EvalConfig, sizes=((2, 2), (3, 3), (4, 4), (5, 5)), **kw) -> dict:
    """-> {"2v2": EvalResult, ...}."""
    raise NotImplementedError


def save_eval(result: EvalResult, path: str) -> None:
    """JSON (contract §11.2)."""
    raise NotImplementedError


def load_eval(path: str) -> EvalResult:
    raise NotImplementedError
