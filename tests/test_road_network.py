from pathlib import Path

from shapely.geometry import LineString

from src.road_network import RoadEdge, RoadNetwork, project_point_to_edge
from src.candidate_generation import candidates_for_point
from shapely.strtree import STRtree
import networkx as nx


def _tiny_network() -> RoadNetwork:
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0)}
    graph = nx.DiGraph()
    graph.add_node(0, x=0, y=0)
    graph.add_node(1, x=1, y=0)
    graph.add_node(2, x=1, y=1)
    edges = []
    index = {}
    for i, (s, d) in enumerate(((0, 1), (1, 2))):
        x1, y1 = nodes[s]
        x2, y2 = nodes[d]
        edge = RoadEdge(i, s, d, 1.0, x1, y1, x2, y2, LineString([(x1, y1), (x2, y2)]))
        edges.append(edge)
        index[(s, d)] = edge
        graph.add_edge(s, d, length=1.0, edge_id=i)
    return RoadNetwork(graph, nodes, edges, index, STRtree([e.geometry for e in edges]), {}, {})


def test_project_and_candidates() -> None:
    network = _tiny_network()
    px, py, dist = project_point_to_edge(0.2, 0.1, network.edges[0])
    assert abs(px - 0.2) < 1e-9
    assert abs(py - 0.0) < 1e-9
    assert dist == 0.1
    cands = candidates_for_point(network, 0.2, 0.05, search_radius=0.5, max_candidates=5)
    assert cands
    assert cands[0].edge.src == 0


def test_missing_ground_truth_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    from src.data_loader import parse_reference_routes

    assert parse_reference_routes(path) == []
