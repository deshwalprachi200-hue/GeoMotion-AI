"""HMM + Viterbi map matching (Newson & Krumm style, custom implementation)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from src.candidate_generation import Candidate, generate_trajectory_candidates
from src.nearest_road import MatchRecord, MatchResult
from src.preprocessing import PreprocessedTrajectory
from src.road_network import RoadNetwork, shortest_path_length
from src.utils import euclidean


@dataclass
class HmmParameters:
    search_radius: float = 1.0
    max_candidates: int = 8
    gps_sigma: float = 0.25
    transition_beta: float = 2.0
    route_distance_cutoff_factor: float = 8.0
    unmatched_penalty: float = 1.0e9


def _log_emission(distance: float, sigma: float) -> float:
    sigma = max(sigma, 1e-9)
    return -0.5 * (distance / sigma) ** 2 - math.log(sigma * math.sqrt(2.0 * math.pi))


def _log_transition(
    network: RoadNetwork,
    prev: Candidate,
    curr: Candidate,
    obs_distance: float,
    beta: float,
    cutoff: float | None,
    route_cache: dict[tuple[int, int], float | None],
) -> float | None:
    """Return log transition probability, or None if no network route exists.

    ``route_cache`` is a per-trajectory dict mapping (src_node, dst_node) to the
    cached shortest-path length (or None when no path was found within the cutoff).
    Caching avoids repeating the same Dijkstra search when candidate edges share
    the same endpoint nodes, which is common in dense road networks and dramatically
    speeds up the Viterbi step.
    """
    key = (prev.edge.dst, curr.edge.src)
    if key not in route_cache:
        route_cache[key] = shortest_path_length(network, key[0], key[1], cutoff=cutoff)
    route = route_cache[key]

    if route is None:
        if prev.edge.edge_id == curr.edge.edge_id or prev.edge.dst == curr.edge.src:
            route = 0.0
        else:
            return None
    # Add remaining travel on the previous/current segments (approx).
    network_distance = route + 0.5 * prev.edge.length + 0.5 * curr.edge.length
    delta = abs(network_distance - obs_distance)
    beta = max(beta, 1e-9)
    return -delta / beta


def hmm_viterbi_match(
    trajectory: PreprocessedTrajectory,
    network: RoadNetwork,
    params: HmmParameters,
) -> MatchResult:
    started = time.perf_counter()
    notes: list[str] = []
    candidate_sets = generate_trajectory_candidates(
        trajectory,
        network,
        search_radius=params.search_radius,
        max_candidates=params.max_candidates,
    )

    n_obs = len(trajectory.points)
    records: list[MatchRecord] = []
    if n_obs == 0:
        return MatchResult("hmm_viterbi", trajectory.order_index, [], 0.0, params.__dict__, ["empty trajectory"])

    # Per-trajectory route cache: (src_node, dst_node) -> shortest-path length or None.
    # Shared across all time steps so the same node pair is never Dijkstra-searched twice.
    route_cache: dict[tuple[int, int], float | None] = {}

    # Viterbi tables: list[obs] -> list[candidate] log-prob and back-pointer
    scores: list[list[float]] = []
    backptr: list[list[int | None]] = []
    fallback_independent = 0
    unresolved_transitions = 0

    for t, (point, cands) in enumerate(zip(trajectory.points, candidate_sets)):
        if not cands:
            scores.append([])
            backptr.append([])
            continue
        emit = [_log_emission(c.distance, params.gps_sigma) for c in cands]
        if t == 0 or not scores[t - 1]:
            scores.append(emit)
            backptr.append([None] * len(cands))
            if t > 0:
                fallback_independent += 1
                notes.append(f"observation {t}: previous step had no candidates; restarting Viterbi")
            continue

        prev_point = trajectory.points[t - 1]
        obs_dist = euclidean(prev_point.x, prev_point.y, point.x, point.y)
        cutoff = None
        if obs_dist > 0:
            cutoff = obs_dist * params.route_distance_cutoff_factor + params.search_radius * 2
        row_scores: list[float] = []
        row_ptr: list[int | None] = []
        prev_cands = candidate_sets[t - 1]
        for j, cand in enumerate(cands):
            best_score = -params.unmatched_penalty
            best_i: int | None = None
            found_trans = False
            for i, prev in enumerate(prev_cands):
                trans = _log_transition(
                    network,
                    prev,
                    cand,
                    obs_dist,
                    params.transition_beta,
                    cutoff,
                    route_cache,
                )
                if trans is None:
                    continue
                found_trans = True
                score = scores[t - 1][i] + trans + emit[j]
                if score > best_score:
                    best_score = score
                    best_i = i
            if not found_trans:
                unresolved_transitions += 1
                # Documented fallback: emission-only using best previous state.
                prev_best = max(range(len(prev_cands)), key=lambda i: scores[t - 1][i])
                best_score = scores[t - 1][prev_best] + emit[j] - math.log(params.unmatched_penalty)
                best_i = prev_best
            row_scores.append(best_score)
            row_ptr.append(best_i)
        scores.append(row_scores)
        backptr.append(row_ptr)

    # Reconstruct: last observation with candidates
    chosen: list[Candidate | None] = [None] * n_obs
    last_t = next((t for t in range(n_obs - 1, -1, -1) if scores[t]), None)
    if last_t is None:
        notes.append("no observation had road candidates")
    else:
        j = max(range(len(scores[last_t])), key=lambda k: scores[last_t][k])
        t = last_t
        while t >= 0 and scores[t]:
            chosen[t] = candidate_sets[t][j]
            ptr = backptr[t][j]
            t -= 1
            while t >= 0 and not scores[t]:
                t -= 1
            if t < 0 or ptr is None:
                break
            j = ptr

    for point, cand in zip(trajectory.points, chosen):
        if cand is None:
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
                    unresolved_reason="no_candidate_or_broken_viterbi_path",
                )
            )
        else:
            records.append(
                MatchRecord(
                    observation_index=point.sequence_index,
                    gps_x=point.x,
                    gps_y=point.y,
                    matched=True,
                    edge_id=cand.edge.edge_id,
                    src=cand.edge.src,
                    dst=cand.edge.dst,
                    matched_x=cand.px,
                    matched_y=cand.py,
                    distance=cand.distance,
                )
            )

    notes.append(f"unresolved_or_fallback_transitions={unresolved_transitions}")
    notes.append(f"viterbi_restarts={fallback_independent}")
    notes.append(f"route_cache_entries={len(route_cache)}")
    return MatchResult(
        method="hmm_viterbi",
        order_index=trajectory.order_index,
        records=records,
        runtime_sec=time.perf_counter() - started,
        parameters=params.__dict__.copy(),
        notes=notes,
    )


def parameters_from_settings(settings: dict) -> HmmParameters:
    block = settings["matching"]
    return HmmParameters(
        search_radius=float(block["search_radius"]),
        max_candidates=int(block["max_candidates"]),
        gps_sigma=float(block["gps_sigma"]),
        transition_beta=float(block["transition_beta"]),
        route_distance_cutoff_factor=float(block["route_distance_cutoff_factor"]),
        unmatched_penalty=float(block["unmatched_penalty"]),
    )
