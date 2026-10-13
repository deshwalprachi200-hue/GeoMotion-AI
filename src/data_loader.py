"""Load SparseTaxi GPS files, reference routes and basic record validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils import load_settings, resolve_data_path


@dataclass
class GpsPoint:
    timestamp: float
    x: float
    y: float
    sequence_index: int
    source_line: int
    timestamp_is_placeholder: bool = True


@dataclass
class Trajectory:
    order_index: int
    file_trajectory_id: int
    car_id: int
    declared_point_count: int
    points: list[GpsPoint]
    source_file: str
    issues: list[str] = field(default_factory=list)

    @property
    def actual_point_count(self) -> int:
        return len(self.points)


@dataclass
class ReferenceRoute:
    order_index: int
    node_ids: list[int]
    source_file: str


def _is_header_line(parts: list[str]) -> bool:
    if len(parts) != 3:
        return False
    return all("." not in token for token in parts)


def parse_coor_file(path: Path, timestamps_are_placeholders: bool = True) -> tuple[int, list[Trajectory]]:
    """Parse a SparseTaxi *.coor.txt file.

    First line is the declared trajectory count. Each trajectory starts with
    ``trajectory_id car_id declared_point_count`` (all integers, no decimal
    point). Coordinate rows are ``timestamp x y``. File trajectory IDs may be
    placeholders; ``order_index`` is the reliable alignment key.
    """
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not raw_lines:
        raise ValueError(f"Empty trajectory file: {path}")

    declared_count = int(raw_lines[0].split()[0])
    trajectories: list[Trajectory] = []
    line_no = 1  # 0-based index into raw_lines; line 0 is the count
    order_index = 0

    while line_no < len(raw_lines):
        stripped = raw_lines[line_no].strip()
        if not stripped:
            line_no += 1
            continue
        parts = stripped.split()
        if not _is_header_line(parts):
            raise ValueError(
                f"{path}:{line_no + 1} expected trajectory header, got {stripped!r}"
            )
        file_tid, car_id, declared_points = (int(parts[0]), int(parts[1]), int(parts[2]))
        line_no += 1
        points: list[GpsPoint] = []
        issues: list[str] = []
        while line_no < len(raw_lines) and len(points) < declared_points:
            row = raw_lines[line_no].strip()
            if not row:
                line_no += 1
                continue
            tokens = row.split()
            if _is_header_line(tokens):
                issues.append(
                    f"declared {declared_points} points but found {len(points)} before next header"
                )
                break
            if len(tokens) != 3:
                issues.append(f"malformed coordinate row at line {line_no + 1}: {row!r}")
                line_no += 1
                continue
            try:
                timestamp, x, y = float(tokens[0]), float(tokens[1]), float(tokens[2])
            except ValueError:
                issues.append(f"non-numeric coordinate row at line {line_no + 1}: {row!r}")
                line_no += 1
                continue
            points.append(
                GpsPoint(
                    timestamp=timestamp,
                    x=x,
                    y=y,
                    sequence_index=len(points),
                    source_line=line_no + 1,
                    timestamp_is_placeholder=timestamps_are_placeholders,
                )
            )
            line_no += 1
        if len(points) != declared_points and not any("declared" in item for item in issues):
            issues.append(
                f"declared {declared_points} points, parsed {len(points)}"
            )
        trajectories.append(
            Trajectory(
                order_index=order_index,
                file_trajectory_id=file_tid,
                car_id=car_id,
                declared_point_count=declared_points,
                points=points,
                source_file=str(path),
                issues=issues,
            )
        )
        order_index += 1

    return declared_count, trajectories


def parse_reference_routes(path: Path) -> list[ReferenceRoute]:
    """Each non-empty row is a sequence of identifiers (node IDs in this dataset)."""
    routes: list[ReferenceRoute] = []
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.strip():
            continue
        node_ids = [int(token) for token in line.split()]
        routes.append(ReferenceRoute(order_index=index, node_ids=node_ids, source_file=str(path)))
    return routes


def load_project_datasets(settings: dict | None = None) -> dict:
    settings = settings or load_settings()
    placeholders = bool(settings["preprocessing"]["treat_timestamps_as_placeholders"])
    sparse_declared, sparse = parse_coor_file(
        resolve_data_path(settings["data"]["sparse_coor"]), placeholders
    )
    dense_declared, dense = parse_coor_file(
        resolve_data_path(settings["data"]["dense_coor"]), placeholders
    )
    references = parse_reference_routes(resolve_data_path(settings["data"]["reference"]))
    return {
        "sparse_declared_count": sparse_declared,
        "dense_declared_count": dense_declared,
        "sparse": sparse,
        "dense": dense,
        "references": references,
    }
