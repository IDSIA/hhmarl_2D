"""Paper-style figures (owner: agent D).  Reads runs/*/metrics.jsonl and eval JSONs only."""
from __future__ import annotations


def smooth_tb(values, weight: float = 0.6):
    """TensorBoard EMA smoothing."""
    raise NotImplementedError


def plot_learning_curves(run_dirs: dict, x: str = "env_steps", y: str = "episode_reward_mean",
                         out_path: str | None = None):
    """Fig 6 / 8a / 10a style: {label: run_dir} -> matplotlib Figure."""
    raise NotImplementedError


def plot_kill_counts(result, out_path: str | None = None):
    """Fig 7a bar chart: k-1, k-2, d-1, d-2, fk-1, fk-2, Draw."""
    raise NotImplementedError


def plot_wld_sweep(results: dict, kind: str = "fight", out_path: str | None = None):
    """Fig 7b / 10b (win/loss/draw) or Fig 8b (escaped/killed/kills) vs team size."""
    raise NotImplementedError
