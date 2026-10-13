"""Shared paths, config loading and small helpers."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("geomotion")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    settings_path = path or project_root() / "config" / "settings.yaml"
    with settings_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_data_path(relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return project_root() / path


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def heading_degrees(ax: float, ay: float, bx: float, by: float) -> float | None:
    if ax == bx and ay == by:
        return None
    return math.degrees(math.atan2(by - ay, bx - ax)) % 360.0


def angular_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
