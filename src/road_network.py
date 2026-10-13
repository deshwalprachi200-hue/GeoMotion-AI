"""Load the SparseTaxi road graph from vertexMap + adjacentList."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from src.utils import LOGGER, ensure_dir, load_settings, resolve_data_path


@dataclass
class RoadEdge:
    edge_id: int
    src: int
    dst: int
    length: float
    x1: float
    y1: float
    x2: float
    y2: float
    geometry: LineString
    highway_type: str | None = None
    is_one_way: bool | None = None
    name: str | None = None
    speed_limit: float | None = None

    @property
    def heading(self) -> float:
        from src.utils import heading_degrees

        value = heading_degrees(self.x1, self.y1, self.x2, self.y2)
        return 0.0 if value is None else value


@dataclass
class RoadNetwork:
    graph: nx.DiGraph
    nodes: dict[int, tuple[float, float]]
    edges: list[RoadEdge]
    edge_index: dict[tuple[int, int], RoadEdge]
    spatial_index: STRtree
    osm_id_to_node: dict[int, int]
    diagnostics: dict = field(default_factory=dict)

    def extent(self) -> dict[str, float]:
        xs = [xy[0] for xy in self.nodes.values()]
        ys = [xy[1] for xy in self.nodes.values()]
        return {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
        }


def load_vertices(path: Path) -> tuple[dict[int, tuple[float, float]], dict[int, int]]:
    nodes: dict[int, tuple[float, float]] = {}
    osm_to_node: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        node_id, osm_id = int(parts[0]), int(parts[1])
        x, y = float(parts[2]), float(parts[3])
        nodes[node_id] = (x, y)
        osm_to_node[osm_id] = node_id
    return nodes, osm_to_node


def load_adjacencies(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    declared = int(lines[0].split()[0])
    rows: list[tuple[int, int, float]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
    return declared, rows


def _parse_highway_line(line: str) -> dict:
    parts = line.split("\t")
    while len(parts) < 5:
        parts.append("")
    speed_raw = parts[4].strip()
    try:
        speed = float(speed_raw) if speed_raw else None
    except ValueError:
        speed = None
    one_way_raw = parts[2].strip()
    is_one_way: bool | None
    if one_way_raw in {"1", "true", "True"}:
        is_one_way = True
    elif one_way_raw in {"0", "false", "False"}:
        is_one_way = False
    else:
        is_one_way = None
    return {
        "n_nodes": parts[0].strip(),
        "highway_type": parts[1].strip() or None,
        "is_one_way": is_one_way,
        "name": parts[3].strip() or None,
        "speed_limit": speed,
    }


def attach_osm_attributes(
    edge_index: dict[tuple[int, int], RoadEdge],
    osm_to_node: dict[int, int],
    waynodes_path: Path,
    highway_path: Path,
) -> dict:
    """Join OSM highway metadata onto graph edges when node sequences match.

    Geometry is NOT replaced by OSM polylines. Edge shape stays the
    vertex-to-vertex segment justified by adjacentList + vertexMap.
    """
    way_lines = waynodes_path.read_text(encoding="utf-8", errors="replace").splitlines()
    highway_lines = highway_path.read_text(encoding="utf-8", errors="replace").splitlines()
    matched = 0
    n_ways = min(len(way_lines), len(highway_lines))
    for index in range(n_ways):
        osm_nodes = [int(tok) for tok in way_lines[index].split() if tok.strip()]
        graph_nodes = [osm_to_node[oid] for oid in osm_nodes if oid in osm_to_node]
        if len(graph_nodes) < 2:
            continue
        attrs = _parse_highway_line(highway_lines[index])
        for src, dst in zip(graph_nodes, graph_nodes[1:]):
            for pair in ((src, dst), (dst, src)):
                edge = edge_index.get(pair)
                if edge is None or edge.highway_type is not None:
                    continue
                edge.highway_type = attrs["highway_type"]
                edge.is_one_way = attrs["is_one_way"]
                edge.name = attrs["name"]
                edge.speed_limit = attrs["speed_limit"]
                matched += 1
    return {"osm_ways_read": n_ways, "edge_attribute_assignments": matched}


def build_road_network(settings: dict | None = None, attach_osm: bool = True) -> RoadNetwork:
    settings = settings or load_settings()
    vertex_path = resolve_data_path(settings["data"]["vertex_map"])
    adj_path = resolve_data_path(settings["data"]["adjacent_list"])
    nodes, osm_to_node = load_vertices(vertex_path)
    declared, rows = load_adjacencies(adj_path)

    graph = nx.DiGraph()
    for node_id, (x, y) in nodes.items():
        graph.add_node(node_id, x=x, y=y)

    edges: list[RoadEdge] = []
    edge_index: dict[tuple[int, int], RoadEdge] = {}
    skipped_missing_node = 0
    duplicate_pairs = 0
    for src, dst, length in rows:
        if src not in nodes or dst not in nodes:
            skipped_missing_node += 1
            continue
        if (src, dst) in edge_index:
            duplicate_pairs += 1
            continue
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        geometry = LineString([(x1, y1), (x2, y2)])
        edge = RoadEdge(
            edge_id=len(edges),
            src=src,
            dst=dst,
            length=length if length > 0 else geometry.length,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            geometry=geometry,
        )
        edges.append(edge)
        edge_index[(src, dst)] = edge
        graph.add_edge(src, dst, length=edge.length, edge_id=edge.edge_id)

    osm_stats = {}
    if attach_osm:
        osm_stats = attach_osm_attributes(
            edge_index,
            osm_to_node,
            resolve_data_path(settings["data"]["waynodes"]),
            resolve_data_path(settings["data"]["highway"]),
        )

    spatial_index = STRtree([edge.geometry for edge in edges])
    n_weak = nx.number_weakly_connected_components(graph)
    n_strong = nx.number_strongly_connected_components(graph)
    diagnostics = {
        "vertex_file": str(vertex_path),
        "adjacency_file": str(adj_path),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "declared_first_line_adjacentList": declared,
        "declared_vs_actual_note": (
            "adjacentList.txt first line equals the vertex count (32552), not the "
            "number of adjacency rows. Actual directed edges are the data rows."
        ),
        "skipped_missing_node": skipped_missing_node,
        "duplicate_directed_pairs_ignored": duplicate_pairs,
        "weakly_connected_components": n_weak,
        "strongly_connected_components": n_strong,
        "geometry_source": "straight LineString from vertex coordinates of src/dst",
        "crs": settings["coordinate_system"],
        "osm_join": osm_stats,
    }
    LOGGER.info("Road network: %s nodes, %s edges", len(nodes), len(edges))
    return RoadNetwork(
        graph=graph,
        nodes=nodes,
        edges=edges,
        edge_index=edge_index,
        spatial_index=spatial_index,
        osm_id_to_node=osm_to_node,
        diagnostics=diagnostics,
    )


def cache_path(settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    return resolve_data_path(settings["data"]["processed_dir"]) / "road_network.pkl"


def load_or_build_network(settings: dict | None = None, attach_osm: bool = True) -> RoadNetwork:
    settings = settings or load_settings()
    path = cache_path(settings)
    if path.exists():
        with path.open("rb") as handle:
            network = pickle.load(handle)
        if isinstance(network, RoadNetwork):
            network.spatial_index = STRtree([edge.geometry for edge in network.edges])
            return network
    network = build_road_network(settings, attach_osm=attach_osm)
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return network


def shortest_path_length(network: RoadNetwork, src: int, dst: int, cutoff: float | None = None) -> float | None:
    if src == dst:
        return 0.0
    if src not in network.graph or dst not in network.graph:
        return None
    try:
        lengths = nx.single_source_dijkstra_path_length(
            network.graph, source=src, cutoff=cutoff, weight="length"
        )
    except nx.NetworkXError:
        return None
    value = lengths.get(dst)
    return None if value is None else float(value)


def project_point_to_edge(x: float, y: float, edge: RoadEdge) -> tuple[float, float, float]:
    """Return (px, py, distance) for the closest point on the edge geometry."""
    point = Point(x, y)
    distance = float(point.distance(edge.geometry))
    projected = edge.geometry.interpolate(edge.geometry.project(point))
    return float(projected.x), float(projected.y), distance
