"""Metrics logging (owner: agent B).  docs/contract.md §9."""
from __future__ import annotations


class MetricsLogger:
    """Appends one JSON object per line to {run_dir}/metrics.jsonl; writes config.json once."""

    def __init__(self, run_dir: str, config: dict):
        raise NotImplementedError

    def log(self, record: dict) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def read_metrics(run_dir: str) -> list[dict]:
    """Read metrics.jsonl (used by notebooks / viz)."""
    raise NotImplementedError
