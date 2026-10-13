from pathlib import Path

from src.data_loader import parse_coor_file, parse_reference_routes


def test_parse_valid_and_short_trajectory(tmp_path: Path) -> None:
    path = tmp_path / "sample.coor.txt"
    path.write_text(
        "\n".join(
            [
                "2",
                "0 0 3",
                "0 10.0 20.0",
                "1 10.5 20.5",
                "2 11.0 21.0",
                "1 0 4",
                "0 1.0 2.0",
                "1 1.1 2.1",
                "2 1.2 2.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    declared, trajs = parse_coor_file(path)
    assert declared == 2
    assert len(trajs) == 2
    assert trajs[0].actual_point_count == 3
    assert trajs[1].declared_point_count == 4
    assert trajs[1].actual_point_count == 3
    assert any("declared 4" in msg for msg in trajs[1].issues)


def test_malformed_point_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "bad.coor.txt"
    path.write_text("1\n0 0 2\n0 1.0 2.0\nnot-a-point\n1 3.0 4.0\n", encoding="utf-8")
    _, trajs = parse_coor_file(path)
    assert trajs[0].issues
    assert trajs[0].actual_point_count == 2


def test_reference_nodes(tmp_path: Path) -> None:
    path = tmp_path / "ref.txt"
    path.write_text("10 11 12\n20 21\n", encoding="utf-8")
    routes = parse_reference_routes(path)
    assert routes[0].node_ids == [10, 11, 12]
    assert routes[1].order_index == 1
