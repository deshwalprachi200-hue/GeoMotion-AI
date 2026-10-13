# GeoMotion AI — selected SparseTaxi dataset files

This package contains the selected SparseTaxi files for the GeoMotion AI prototype.

## Included
- `manual_matched/beijing_sparse.coor.txt`: 23 manually matched GPS trajectories (reference/dense trajectories).
- `manual_matched/beijing_sparse.txt`: 23 reference route sequences.
- `sparse_trajectories/heavyTraj_sparse.coor.txt`: corresponding sparse/noisy input trajectories for experiments.
- `map/vertexMap.txt`, `map/adjacentList.txt`, and `map/osm_data/*`: road-network nodes, directed adjacency, and OSM metadata/geometry.

## Important parsing notes
- In `.coor.txt`, the first line is the declared trajectory count. Each trajectory header is `trajectory_id car_id declared_point_count`; each following point is `timestamp x y`.
- The repository notes that IDs/timestamps may be placeholders and that some trajectories are missing.
- For the selected `manual_matched/beijing_sparse.coor.txt`, three trajectories have fewer actual coordinate rows than the declared point count (trajectory IDs 2, 12, and 16 are short by 1, 1, and 3 points respectively). Do not invent points; validate actual rows and record this in preprocessing.
- The corresponding reference route rows use identifiers that empirically match `vertexMap.txt` node IDs, despite the upstream README describing them as edge IDs. Validate each identifier against `vertexMap.txt`; convert consecutive route nodes to directed edges using `adjacentList.txt` where possible.
- GPS coordinates are stored as x=latitude, y=longitude in these files. Verify map alignment during preprocessing.
- Do not claim model accuracy until a held-out evaluation has been run.

## Source
SparseTaxi repository: https://github.com/yangli-learning/SparseTaxi
