"""Nearby road candidates for each GPS observation."""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Point

from src.preprocessing import PreprocessedTrajectory
from src.road_network import RoadEdge, RoadNetwork, project_point_to_edge
from src.utils import angular_difference, heading_degrees


@dataclass
class Candidate:
    edge: RoadEdge
    distance: float
    px: float
    py: float
    heading_difference: float | None
    gps_heading: float | None


def candidates_for_point(
    network: RoadNetwork,
    x: float,
    y: float,
    *,
    search_radius: float,
    max_candidates: int,
    gps_heading: float | None = None,
) -> list[Candidate]:
    point = Point(x, y)
    tree = network.spatial_index
    try:
        indices = list(tree.query(point, predicate="dwithin", distance=search_radius))
    except Exception:
        envelope = point.buffer(search_radius)
        indices = list(tree.query(envelope))

    found: list[Candidate] = []
    for idx in indices:
        edge: RoadEdge = network.edges[int(idx)]
        px, py, distance = project_point_to_edge(x, y, edge)
        if distance > search_radius:
            continue
        heading_diff = angular_difference(gps_heading, edge.heading)
        found.append(
            Candidate(
                edge=edge,
                distance=distance,
                px=px,
                py=py,
                heading_difference=heading_diff,
                gps_heading=gps_heading,
            )
        )
    found.sort(key=lambda item: item.distance)
    return found[:max_candidates]


def generate_trajectory_candidates(
    trajectory: PreprocessedTrajectory,
    network: RoadNetwork,
    *,
    search_radius: float,
    max_candidates: int,
) -> list[list[Candidate]]:
    per_point: list[list[Candidate]] = []
    for index, point in enumerate(trajectory.points):
        gps_heading = point.heading
        if gps_heading is None and index + 1 < len(trajectory.points):
            nxt = trajectory.points[index + 1]
            gps_heading = heading_degrees(point.x, point.y, nxt.x, nxt.y)
        per_point.append(
            candidates_for_point(
                network,
                point.x,
                point.y,
                search_radius=search_radius,
                max_candidates=max_candidates,
                gps_heading=gps_heading,
            )
        )
    return per_point
