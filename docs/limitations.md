# Limitations

1. **Unknown CRS.** Distances are native units. Do not report them as metres on a slide unless you recompute with a proven projection.
2. **Placeholder timestamps and IDs.** Sparse `trajectory_id` values repeat. Always use `order_index` 0–22.
3. **Three dense trajectories are short** (order_index 2, 12, 16). Missing rows were not filled in.
4. **Two reference paths** have one consecutive node pair that is not a graph edge.
5. **adjacentList first line** equals the vertex count, not the edge-row count.
6. **Edge F1 is a weak accuracy proxy** because sparse GPS is much shorter than the reference node list.
7. **No supervised model** was trained; labels are not point-wise.
8. **One-way OSM flags** are attributes only; driving constraints actually used are the directed `adjacentList` edges.
9. **Geometry** of each graph edge is a straight line between vertices, not the full OSM polyline.
10. **Folium OSM world tiles would be wrong** if enabled; the dashboard uses projected/Simple views.
11. **HMM parameters** (`sigma`, `beta`, radius) are not cross-validated on a held-out city.
12. Default experiments run a **sample** of trajectories (`demo.sample_mode`). Use `--full` for all 23.
