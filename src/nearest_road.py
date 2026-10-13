"""Independent nearest-road baseline (one observation at a time)."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from src.candidate_generation import Candidate, generate_trajectory_candidates
from src.preprocessing import PreprocessedTrajectory
from src.road_network import RoadNetwork


@dataclass
class MatchRecord:
    observation_index: int
    gps_x: float
    gps_y: float
    matched: bool
    edge_id: int | None
    src: int | None
    dst: int | None
    matched_x: float | None
    matched_y: float | None
    distance: float | None
    unresolved_reason: str | None = None


@dataclass
class MatchResult:
    method: str
    order_index: int
    records: list[MatchRecord]
    runtime_sec: float
    parameters: dict
    notes: list[str] = field(default_factory=list)

    @property
    def matched_count(self) -> int:
        return sum(1 for rec in self.records if rec.matched)

    @property
    def unmatched_count(self) -> int:
        return sum(1 for rec in self.records if not rec.matched)


def nearest_road_match(
    trajectory: PreprocessedTrajectory,
    network: RoadNetwork,
    *,
    search_radius: float,
    max_candidates: int,
) -> MatchResult:
    started = time.perf_counter()
    candidate_sets = generate_trajectory_candidates(
        trajectory,
        network,
        search_radius=search_radius,
        max_candidates=max_candidates,
    )
    records: list[MatchRecord] = []
    notes: list[str] = []
    for point, candidates in zip(trajectory.points, candidate_sets):
        if not candidates:
            records.append(
                MatchRecord(
                    observation_index=point.sequence_index,
                    gps_x=point.x,
                    gps_y=point.y,
                    matched=False,
                    edge_id=None,
                    src=None,
                    dst=None,
                    matched_x=None,
                    matched_y=None,
                    distance=None,
                    unresolved_reason="no_candidate_within_radius",
                )
            )
            continue
        best: Candidate = candidates[0]
        records.append(
            MatchRecord(
                observation_index=point.sequence_index,
                gps_x=point.x,
                gps_y=point.y,
                matched=True,
                edge_id=best.edge.edge_id,
                src=best.edge.src,
                dst=best.edge.dst,
                matched_x=best.px,
                matched_y=best.py,
                distance=best.distance,
            )
        )
    if any(not rec.matched for rec in records):
        notes.append("some observations had no road within the search radius")
    return MatchResult(
        method="nearest_road",
        order_index=trajectory.order_index,
        records=records,
        runtime_sec=time.perf_counter() - started,
        parameters={"search_radius": search_radius, "max_candidates": max_candidates},
        notes=notes,
    )
