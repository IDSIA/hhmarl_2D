"""Trajectory renders and animations (owner: agent D).  Pure matplotlib (no cairo/cartopy)."""
from __future__ import annotations


def render_episode(rec, out_path: str | None = None, upto: int | None = None):
    """Static trajectory plot of an EpisodeRecord (Fig 9 style)."""
    raise NotImplementedError


def animate_episode(rec, out_path: str, fps: int = 20):
    """GIF/MP4 animation (matplotlib.animation; PillowWriter fallback)."""
    raise NotImplementedError
