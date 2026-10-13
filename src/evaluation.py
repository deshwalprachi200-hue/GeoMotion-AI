"""Compare matched routes with verified reference node sequences."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data_loader import ReferenceRoute
from src.nearest_road import MatchResult
from src.road_network import RoadNetwork


@dataclass
class RouteEval:
    order_index: int
    method: str
    n_observations: int
    matched_count: int
    unmatched_count: int
    mean_match_distance: float | None
    median_match_distance: float | None
    edge_precision: float | None
    edge_recall: float | None
    edge_f1: float | None
    reference_edge_count: int
    predicted_edge_count: int
    disconnected_reference_pairs: int
    runtime_sec: float
    notes: list[str] = field(default_factory=list)


def reference_edges(
    route: ReferenceRoute, network: RoadNetwork
) -> tuple[set[tuple[int, int]], int, list[str]]:
    """Convert consecutive reference node IDs to directed graph edges.

    Identifiers were checked against vertexMap node IDs. They are not treated
    as interchangeable with sequential adjacency-row indices.
    """
    notes: list[str] = []
    edges: set[tuple[int, int]] = set()
    disconnected = 0
    missing_nodes = 0
    for src, dst in zip(route.node_ids, route.node_ids[1:]):
        if src not in network.nodes:
            missing_nodes += 1
            continue
        if dst not in network.nodes:
            missing_nodes += 1
            continue
        if (src, dst) in network.edge_index:
            edges.add((src, dst))
        elif (dst, src) in network.edge_index:
            notes.append(f"used reverse edge ({dst},{src}) for consecutive nodes ({src},{dst})")
            edges.add((dst, src))
        else:
            disconnected += 1
    if missing_nodes:
        notes.append(f"{missing_nodes} reference identifiers were not graph nodes")
    if disconnected:
        notes.append(f"{disconnected} consecutive node pairs are not connected by an edge")
    return edges, disconnected, notes


def predicted_edges(result: MatchResult) -> set[tuple[int, int]]:
    return {
        (rec.src, rec.dst)
        for rec in result.records
        if rec.matched and rec.src is not None and rec.dst is not None
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def evaluate_result(
    result: MatchResult,
    route: ReferenceRoute | None,
    network: RoadNetwork,
) -> RouteEval:
    distances = [rec.distance for rec in result.records if rec.matched and rec.distance is not None]
    notes = list(result.notes)
    precision = recall = f1 = None
    ref_count = 0
    pred_count = 0
    disconnected = 0
    if route is None:
        notes.append("no reference route for this trajectory index; ground-truth edge metrics omitted")
    else:
        ref_set, disconnected, ref_notes = reference_edges(route, network)
        notes.extend(ref_notes)
        pred_set = predicted_edges(result)
        ref_count = len(ref_set)
        pred_count = len(pred_set)
        if ref_count == 0:
            notes.append("reference produced no connected edges; skip precision/recall")
        else:
            overlap = len(ref_set & pred_set)
            precision = overlap / pred_count if pred_count else 0.0
            recall = overlap / ref_count
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            notes.append(
                "edge precision/recall compare matched observation edges to the "
                "reference node-path edges. Sparse GPS has fewer points than the "
                "reference path, so recall is expected to be low."
            )
    return RouteEval(
        order_index=result.order_index,
        method=result.method,
        n_observations=len(result.records),
        matched_count=result.matched_count,
        unmatched_count=result.unmatched_count,
        mean_match_distance=_mean(distances),
        median_match_distance=_median(distances),
        edge_precision=precision,
        edge_recall=recall,
        edge_f1=f1,
        reference_edge_count=ref_count,
        predicted_edge_count=pred_count,
        disconnected_reference_pairs=disconnected,
        runtime_sec=result.runtime_sec,
        notes=notes,
    )


def eval_to_row(item: RouteEval) -> dict:
    return {
        "order_index": item.order_index,
        "method": item.method,
        "n_observations": item.n_observations,
        "matched_count": item.matched_count,
        "unmatched_count": item.unmatched_count,
        "mean_match_distance": item.mean_match_distance,
        "median_match_distance": item.median_match_distance,
        "edge_precision": item.edge_precision,
        "edge_recall": item.edge_recall,
        "edge_f1": item.edge_f1,
        "reference_edge_count": item.reference_edge_count,
        "predicted_edge_count": item.predicted_edge_count,
        "disconnected_reference_pairs": item.disconnected_reference_pairs,
        "runtime_sec": item.runtime_sec,
    }
