# Dataset notes (from actual files)

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
