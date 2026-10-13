"""GPS cleaning that records every rejection instead of dropping data silently."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data_loader import GpsPoint, Trajectory
from src.utils import euclidean, heading_degrees


@dataclass
class CleanedPoint:
    timestamp: float
    x: float
    y: float
    sequence_index: int
    dt: float | None
    displacement: float | None
    speed: float | None
    heading: float | None
    implausible_jump: bool = False


@dataclass
class PreprocessedTrajectory:
    order_index: int
    file_trajectory_id: int
    source: str
    points: list[CleanedPoint]
    removed: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def preprocess_trajectory(
    trajectory: Trajectory,
    *,
    drop_duplicates: bool = True,
    duplicate_xy_decimals: int = 6,
    sort_by_timestamp: bool = True,
    jump_threshold: float = 8.0,
    timestamps_are_placeholders: bool = True,
) -> PreprocessedTrajectory:
    removed: list[dict] = []
    warnings = list(trajectory.issues)
    valid: list[GpsPoint] = []
    for point in trajectory.points:
        if not (_finite(point.x) and _finite(point.y)):
            removed.append({"reason": "non_finite_coordinate", "line": point.source_line})
            continue
        valid.append(point)

    if sort_by_timestamp and valid:
        timestamps = [p.timestamp for p in valid]
        if timestamps != sorted(timestamps):
            warnings.append("timestamps were not sorted; sorted for processing")
            valid = sorted(valid, key=lambda item: (item.timestamp, item.sequence_index))

    if timestamps_are_placeholders:
        warnings.append(
            "timestamps are treated as placeholders, not wall-clock time "
            "(SparseTaxi README: IDs/timestamps may be placeholders)"
        )

    cleaned: list[CleanedPoint] = []
    seen: set[tuple[float, float]] = set()
    previous: CleanedPoint | None = None
    for original in valid:
        key = (round(original.x, duplicate_xy_decimals), round(original.y, duplicate_xy_decimals))
        if drop_duplicates and key in seen:
            removed.append(
                {
                    "reason": "duplicate_xy",
                    "line": original.source_line,
                    "x": original.x,
                    "y": original.y,
                }
            )
            continue
        seen.add(key)
        dt = None
        displacement = None
        speed = None
        heading = None
        jump = False
        if previous is not None:
            dt = original.timestamp - previous.timestamp
            displacement = euclidean(previous.x, previous.y, original.x, original.y)
            if dt and dt > 0 and not timestamps_are_placeholders:
                speed = displacement / dt
            heading = heading_degrees(previous.x, previous.y, original.x, original.y)
            if displacement is not None and displacement > jump_threshold:
                jump = True
        item = CleanedPoint(
            timestamp=original.timestamp,
            x=original.x,
            y=original.y,
            sequence_index=len(cleaned),
            dt=dt,
            displacement=displacement,
            speed=speed,
            heading=heading,
            implausible_jump=jump,
        )
        cleaned.append(item)
        previous = item

    if not cleaned:
        warnings.append("trajectory has no valid points after cleaning")

    return PreprocessedTrajectory(
        order_index=trajectory.order_index,
        file_trajectory_id=trajectory.file_trajectory_id,
        source=trajectory.source_file,
        points=cleaned,
        removed=removed,
        warnings=warnings,
    )


def preprocess_all(trajectories: list[Trajectory], settings: dict) -> list[PreprocessedTrajectory]:
    prep = settings["preprocessing"]
    return [
        preprocess_trajectory(
            trajectory,
            drop_duplicates=prep["drop_duplicates"],
            duplicate_xy_decimals=prep["duplicate_xy_decimals"],
            sort_by_timestamp=prep["sort_by_timestamp"],
            jump_threshold=prep["jump_threshold"],
            timestamps_are_placeholders=prep["treat_timestamps_as_placeholders"],
        )
        for trajectory in trajectories
    ]
