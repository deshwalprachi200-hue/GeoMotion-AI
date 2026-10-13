from src.data_loader import ReferenceRoute
from src.evaluation import evaluate_result, reference_edges
from src.nearest_road import MatchRecord, MatchResult
from tests.test_map_matching import _network


def test_metrics_without_ground_truth() -> None:
    result = MatchResult(
        method="nearest_road",
        order_index=0,
        records=[
            MatchRecord(0, 0, 0, True, 0, 0, 1, 0.0, 0.0, 0.1),
            MatchRecord(1, 1, 0, False, None, None, None, None, None, None, "no_candidate"),
        ],
        runtime_sec=0.01,
        parameters={},
    )
    ev = evaluate_result(result, None, _network())
    assert ev.matched_count == 1
    assert ev.unmatched_count == 1
    assert ev.edge_precision is None
    assert "no reference route" in ev.notes[0]


def test_edge_overlap() -> None:
    network = _network()
    route = ReferenceRoute(0, [0, 1, 2], "mem")
    edges, disconnected, _ = reference_edges(route, network)
    assert disconnected == 0
    assert (0, 1) in edges
    result = MatchResult(
        "nearest_road",
        0,
        [
            MatchRecord(0, 0, 0, True, 0, 0, 1, 0, 0, 0.1),
            MatchRecord(1, 2, 0, True, 1, 1, 2, 2, 0, 0.1),
        ],
        0.0,
        {},
    )
    ev = evaluate_result(result, route, network)
    assert ev.edge_precision == 1.0
    assert ev.edge_recall == 1.0


def test_export_rows_shape() -> None:
    from src.visualization import match_export_rows

    result = MatchResult("nearest_road", 0, [MatchRecord(0, 1, 2, False, None, None, None, None, None, None, "x")], 0, {})
    rows = match_export_rows(result)
    assert rows[0]["gps_x"] == 1
    assert rows[0]["matched"] is False
