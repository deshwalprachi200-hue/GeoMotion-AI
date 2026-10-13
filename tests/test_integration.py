"""Integration smoke-test: load real dataset files and run both algorithms on 2 trajectories.

This test is slow (requires loading the full road network from disk) and is
marked with the ``integration`` custom marker so it can be excluded from the
normal fast test run:

    pytest -m "not integration"        # fast unit tests only
    pytest -m integration              # only integration tests
    pytest                             # all tests (default, will be slow)
"""

from __future__ import annotations

import pytest

from src.data_loader import load_project_datasets
from src.hmm_map_matching import HmmParameters, hmm_viterbi_match
from src.nearest_road import nearest_road_match
from src.preprocessing import preprocess_all
from src.road_network import build_road_network
from src.utils import load_settings


@pytest.mark.integration
def test_full_pipeline_on_real_data() -> None:
    """Load real files, build road network, run both algorithms on first 2 trajectories."""
    settings = load_settings()
    bundle = load_project_datasets(settings)
    network = build_road_network(settings, attach_osm=False)  # skip OSM join for speed

    sparse = preprocess_all(bundle["sparse"], settings)
    sample = [t for t in sparse if len(t.points) >= 2][:2]

    params = HmmParameters(search_radius=1.0, max_candidates=5)

    for traj in sample:
        nr = nearest_road_match(traj, network, search_radius=1.0, max_candidates=5)
        assert nr.method == "nearest_road"
        assert nr.order_index == traj.order_index
        assert len(nr.records) == len(traj.points)
        # All points should match (road network overlaps GPS extents)
        assert nr.matched_count > 0, f"traj {traj.order_index}: nearest road matched nothing"

        hmm = hmm_viterbi_match(traj, network, params)
        assert hmm.method == "hmm_viterbi"
        assert len(hmm.records) == len(traj.points)
        assert hmm.matched_count > 0, f"traj {traj.order_index}: HMM matched nothing"
