from src.data_loader import GpsPoint, Trajectory
from src.preprocessing import preprocess_trajectory


def _traj(points: list[tuple[float, float, float]]) -> Trajectory:
    gps = [
        GpsPoint(timestamp=t, x=x, y=y, sequence_index=i, source_line=i + 1)
        for i, (t, x, y) in enumerate(points)
    ]
    return Trajectory(0, 0, 0, len(gps), gps, "mem")


def test_rejects_non_finite_and_duplicates() -> None:
    traj = _traj([(0, 1.0, 1.0), (1, float("nan"), 1.0), (2, 1.0, 1.0), (3, 2.0, 2.0)])
    cleaned = preprocess_trajectory(traj, jump_threshold=100)
    assert len(cleaned.points) == 2
    reasons = {row["reason"] for row in cleaned.removed}
    assert "non_finite_coordinate" in reasons
    assert "duplicate_xy" in reasons


def test_jump_flag() -> None:
    traj = _traj([(0, 0.0, 0.0), (1, 50.0, 0.0)])
    cleaned = preprocess_trajectory(traj, jump_threshold=8.0)
    assert cleaned.points[1].implausible_jump is True
