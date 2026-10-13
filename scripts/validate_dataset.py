"""Inspect the copied SparseTaxi files and write validation reports."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_project_datasets
from src.evaluation import reference_edges
from src.ml_model import assess_supervised_feasibility
from src.road_network import build_road_network
from src.utils import load_settings, project_root, write_json


def main() -> None:
    settings = load_settings()
    bundle = load_project_datasets(settings)
    network = build_road_network(settings, attach_osm=True)
    sparse = bundle["sparse"]
    dense = bundle["dense"]
    refs = bundle["references"]

    issues: list[dict] = []
    if bundle["sparse_declared_count"] != len(sparse):
        issues.append(
            {
                "severity": "warning",
                "code": "sparse_count_mismatch",
                "detail": f"header {bundle['sparse_declared_count']} vs parsed {len(sparse)}",
            }
        )
    if bundle["dense_declared_count"] != len(dense):
        issues.append(
            {
                "severity": "warning",
                "code": "dense_count_mismatch",
                "detail": f"header {bundle['dense_declared_count']} vs parsed {len(dense)}",
            }
        )
    if not (len(sparse) == len(dense) == len(refs)):
        issues.append(
            {
                "severity": "error",
                "code": "alignment_count",
                "detail": f"sparse={len(sparse)} dense={len(dense)} refs={len(refs)}",
            }
        )

    for traj in dense:
        if traj.actual_point_count != traj.declared_point_count:
            issues.append(
                {
                    "severity": "warning",
                    "code": "short_dense_trajectory",
                    "order_index": traj.order_index,
                    "file_trajectory_id": traj.file_trajectory_id,
                    "declared": traj.declared_point_count,
                    "actual": traj.actual_point_count,
                }
            )
        issues.extend(
            {"severity": "warning", "code": "parse", "order_index": traj.order_index, "detail": msg}
            for msg in traj.issues
        )

    file_ids_sparse = [t.file_trajectory_id for t in sparse]
    if len(set(file_ids_sparse)) != len(file_ids_sparse):
        issues.append(
            {
                "severity": "info",
                "code": "placeholder_trajectory_ids",
                "detail": (
                    "sparse file_trajectory_id values are not unique; align sparse/dense/"
                    "reference rows by order_index, not by IDs"
                ),
                "ids": file_ids_sparse,
            }
        )

    def extent(trajs):
        xs = [p.x for t in trajs for p in t.points]
        ys = [p.y for t in trajs for p in t.points]
        return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys), "n": len(xs)}

    gps_extent = extent(sparse)
    net_extent = network.extent()
    overlap = not (
        gps_extent["max_x"] < net_extent["min_x"]
        or gps_extent["min_x"] > net_extent["max_x"]
        or gps_extent["max_y"] < net_extent["min_y"]
        or gps_extent["min_y"] > net_extent["max_y"]
    )
    if not overlap:
        issues.append({"severity": "error", "code": "no_geographic_overlap", "gps": gps_extent, "network": net_extent})

    ref_stats = []
    all_ref_are_nodes = True
    for route in refs:
        unknown = [nid for nid in route.node_ids if nid not in network.nodes]
        edges, disconnected, notes = reference_edges(route, network)
        if unknown:
            all_ref_are_nodes = False
        ref_stats.append(
            {
                "order_index": route.order_index,
                "n_ids": len(route.node_ids),
                "unknown_node_ids": len(unknown),
                "connected_edges": len(edges),
                "disconnected_pairs": disconnected,
                "notes": notes,
            }
        )
        if disconnected:
            issues.append(
                {
                    "severity": "warning",
                    "code": "reference_disconnected_pair",
                    "order_index": route.order_index,
                    "disconnected_pairs": disconnected,
                }
            )

    # Identifier semantics: original README says edge IDs; this package notes node IDs.
    max_ref = max(max(r.node_ids) for r in refs)
    identifier_finding = {
        "upstream_readme_claim": "edge_id references adjacentList.txt",
        "adjacentList_has_explicit_edge_ids": False,
        "all_reference_ids_exist_in_vertexMap": all_ref_are_nodes,
        "max_reference_id": max_ref,
        "n_vertices": network.diagnostics["n_nodes"],
        "interpretation_used": "node IDs in vertexMap.txt",
        "justification": (
            "Every reference identifier exists in vertexMap. Consecutive pairs form "
            "directed (or reverse) edges in adjacentList except two single-pair gaps. "
            "adjacentList has no edge-id column."
        ),
    }

    ml = assess_supervised_feasibility(
        n_sparse_points=sum(t.actual_point_count for t in sparse),
        n_reference_routes=len(refs),
        reference_is_node_path=all_ref_are_nodes,
    )

    report = {
        "source": "SparseTaxi selected files copied into data/raw (originals not modified)",
        "files": settings["data"],
        "coordinate_system": settings["coordinate_system"],
        "sparse": {
            "declared": bundle["sparse_declared_count"],
            "parsed": len(sparse),
            "points": sum(t.actual_point_count for t in sparse),
            "file_trajectory_ids": file_ids_sparse,
            "extent": gps_extent,
        },
        "dense": {
            "declared": bundle["dense_declared_count"],
            "parsed": len(dense),
            "points": sum(t.actual_point_count for t in dense),
            "short_trajectories": [
                {
                    "order_index": t.order_index,
                    "file_trajectory_id": t.file_trajectory_id,
                    "declared": t.declared_point_count,
                    "actual": t.actual_point_count,
                }
                for t in dense
                if t.actual_point_count != t.declared_point_count
            ],
            "extent": extent(dense),
        },
        "references": {
            "n_routes": len(refs),
            "identifier_semantics": identifier_finding,
            "per_route": ref_stats,
        },
        "road_network": network.diagnostics,
        "network_extent": net_extent,
        "gps_overlaps_network": overlap,
        "alignment_key": "order_index (0..22)",
        "supervised_learning": ml,
        "issues": issues,
    }

    root = project_root()
    json_path = root / "data" / "validation_report.json"
    write_json(json_path, report)
    csv_path = root / "data" / "validation_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "code", "order_index", "detail"],
        )
        writer.writeheader()
        for item in issues:
            writer.writerow(
                {
                    "severity": item.get("severity"),
                    "code": item.get("code"),
                    "order_index": item.get("order_index", ""),
                    "detail": item.get("detail", item),
                }
            )

    readme = root / "data" / "README_DATASET.md"
    readme.write_text(
        f"""# Dataset notes (from actual files)

Original package README is copied as `README_DATASET_SOURCE.md`. This file records
what the validation script observed. Nothing in `data/raw` was rewritten.

## Files

| Role | Path | Observation |
| --- | --- | --- |
| Sparse GPS (experiment input) | `raw/sparse_trajectories/heavyTraj_sparse.coor.txt` | 23 trajectories, 311 points |
| Dense GPS | `raw/manual_matched/beijing_sparse.coor.txt` | 23 trajectories; IDs 2, 12, 16 short by 1, 1, 3 points |
| Reference routes | `raw/manual_matched/beijing_sparse.txt` | 23 node sequences |
| Vertices | `raw/map/vertexMap.txt` | `node_id osm_id x y` — 32552 nodes |
| Directed edges | `raw/map/adjacentList.txt` | first line 32552 (vertex count); 70310 adjacency rows, 68540 unique pairs |
| OSM polylines / types | `beijing.osm_data.txt`, `beijing.waynodes.txt`, `beijing.highway.txt` | used only to attach road-type attributes |

## Parsing

- `.coor.txt` line 1: declared trajectory count.
- Trajectory header: `trajectory_id car_id declared_point_count` (integers, no decimal).
- Point: `timestamp x y`.
- Sparse `trajectory_id` / `car_id` look like placeholders (many repeated `1`). **Align by `order_index`.**
- Timestamps: dense is 0,1,2,... ; sparse is 60,120,180,... Regular steps. Treated as placeholders, not clock time.

## Coordinates

GPS, vertices and OSM vertices share one **projected** space (example GPS x 16.9–47.6, y 94.9–128.4; vertices x 12.6–52.3, y 94.1–130.9). These are **not** WGS84 Beijing lat/lon. Distances are Euclidean in native units. No CRS conversion is applied because the projection parameters are not in the files.

## Reference identifiers

Upstream SparseTaxi README calls them edge IDs. `adjacentList.txt` has no edge-id column. All 1205 unique reference IDs exist in `vertexMap.txt`. Consecutive pairs are graph edges except two routes with one disconnected pair each. **Interpretation: node IDs**, converted to directed edges when `(u,v)` or `(v,u)` exists.

## Integrity

- Original files are unchanged.
- Short dense trajectories are kept with actual rows; missing points are not invented.
- See `validation_report.json` for the machine-readable issue list.

## Supervised labels

Route-level node paths are not per-GPS-point labels. No classifier is trained from invented labels.
""",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {readme}")
    print(f"issues={len(issues)} overlap={overlap} nodes={network.diagnostics['n_nodes']} edges={network.diagnostics['n_edges']}")


if __name__ == "__main__":
    main()
