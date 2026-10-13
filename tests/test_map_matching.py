from shapely.geometry import LineString
from shapely.strtree import STRtree
import networkx as nx

from src.hmm_map_matching import HmmParameters, hmm_viterbi_match
from src.nearest_road import nearest_road_match
from src.preprocessing import preprocess_trajectory
from src.data_loader import GpsPoint, Trajectory
from src.road_network import RoadEdge, RoadNetwork


def _network() -> RoadNetwork:
    nodes = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (4.0, 0.0), 3: (2.0, 2.0)}
    pairs = [(0, 1), (1, 2), (1, 3)]
    graph = nx.DiGraph()
    for nid, (x, y) in nodes.items():
        graph.add_node(nid, x=x, y=y)
    edges = []
    index = {}
    for i, (s, d) in enumerate(pairs):
        x1, y1 = nodes[s]
        x2, y2 = nodes[d]
        edge = RoadEdge(i, s, d, abs(x2 - x1) + abs(y2 - y1), x1, y1, x2, y2, LineString([(x1, y1), (x2, y2)]))
        edges.append(edge)
        index[(s, d)] = edge
        graph.add_edge(s, d, length=edge.length, edge_id=i)
    return RoadNetwork(graph, nodes, edges, index, STRtree([e.geometry for e in edges]), {}, {})


def _prep() -> object:
    points = [
        GpsPoint(0, 0.05, 0.04, 0, 1),
        GpsPoint(1, 1.9, 0.05, 1, 2),
        GpsPoint(2, 3.9, 0.02, 2, 3),
    ]
    return preprocess_trajectory(Trajectory(0, 0, 0, 3, points, "mem"), jump_threshold=100)


def test_nearest_road_selects_closest() -> None:
    result = nearest_road_match(_prep(), _network(), search_radius=1.0, max_candidates=5)
    assert result.matched_count == 3
    assert result.records[0].src == 0


def test_hmm_handles_missing_candidates_without_inventing_route() -> None:
    traj = preprocess_trajectory(
        Trajectory(
            0,
            0,
            0,
            2,
            [GpsPoint(0, 0.05, 0.0, 0, 1), GpsPoint(1, 50.0, 50.0, 1, 2)],
            "mem",
        ),
        jump_threshold=1000,
    )
    result = hmm_viterbi_match(traj, _network(), HmmParameters(search_radius=0.5, max_candidates=4))
    assert result.records[0].matched is True
    assert result.records[1].matched is False
    assert result.records[1].unresolved_reason
