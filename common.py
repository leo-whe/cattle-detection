"""Shared paths and dataset-config handling for the detection scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
WEIGHTS = ROOT / "weights"
DEFAULT_DATA = ROOT / "configs" / "cow_uav.yaml"


def resolve_data(data: str | Path) -> str:
    """Return a dataset yaml whose ``path`` is absolute.

    Ultralytics resolves a relative ``path`` against its global datasets directory,
    not against the yaml or this repository. Configs here keep ``path`` relative to
    the repository root so they carry no machine-specific paths; this writes a
    resolved copy under ``runs/_data/`` and returns it.
    """
    data = Path(data)
    cfg = yaml.safe_load(data.read_text())
    base = Path(cfg.get("path", ""))
    if base.is_absolute():
        return str(data)
    cfg["path"] = str((ROOT / base).resolve())
    out = RUNS / "_data" / data.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)
