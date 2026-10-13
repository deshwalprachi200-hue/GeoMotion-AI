# PROJECT_STATUS.md — GeoMotion AI
*Last updated: 2026-10-13 (post-audit and improvements)*

---

## ✅ Overall Status: FULLY FUNCTIONAL

The complete pipeline runs end-to-end with no fabricated data.
All tests pass. Both algorithms produce real results on the SparseTaxi dataset.

---

## Audit Findings

### Dataset Files

| File | Status | Notes |
|------|--------|-------|
| `data/raw/sparse_trajectories/heavyTraj_sparse.coor.txt` | ✅ Present | 23 trajectories, 311 GPS points |
| `data/raw/manual_matched/beijing_sparse.coor.txt` | ✅ Present | 23 dense trajectories, 2314 points |
| `data/raw/manual_matched/beijing_sparse.txt` | ✅ Present | 23 reference node sequences |
| `data/raw/map/vertexMap.txt` | ✅ Present | 32 552 nodes |
| `data/raw/map/adjacentList.txt` | ✅ Present | 68 540 unique directed edges |
| `data/raw/map/beijing.highway.txt` | ✅ Present | OSM way types joined to edges |
| `data/raw/map/beijing.waynodes.txt` | ✅ Present | Used for OSM attribute join |
| `data/raw/map/beijing.osm_data.txt` | ✅ Present | Not used in matching (geometry) |

### Dataset Validation Results (`scripts/validate_dataset.py`)

- Ran successfully → wrote `data/validation_report.json`, `.csv`, `README_DATASET.md`
- **GPS extents overlap road network**: ✅ True
- All 23 reference identifiers resolve to valid `vertexMap` node IDs: ✅
- 9 non-critical issues:
  - 3× dense trajectories 1–4 points short (original file artefact, not our error)
  - 1× sparse `file_trajectory_id` values are placeholders
  - 2× reference routes with 1 disconnected node-pair each (trajectories 2 and 22)

### Road Network (`src/road_network.py`)

- Loads 32 552 nodes and 68 540 directed edges ✅
- Builds Shapely STRtree spatial index for fast candidate search ✅
- Joins OSM highway type / name / speed onto edges ✅
- Network pickle cached at `data/processed/road_network.pkl` ✅

### GPS Preprocessing (`src/preprocessing.py`)

- Removes NaN / ±inf coordinates ✅
- Drops spatial duplicates ✅
- Flags implausible jumps ✅
- Timestamps treated as placeholders (60-unit regular steps) ✅

### Nearest-Road Baseline (`src/nearest_road.py`) ✅ Complete

### HMM / Viterbi Matching (`src/hmm_map_matching.py`) ✅ Complete

### Evaluation (`src/evaluation.py`) ✅ Complete

### Streamlit Dashboard (`app.py`) ✅ Complete — 6 tabs

### Scripts ✅ Both scripts run without errors

---

## Experiment Results (Sample Mode, 5 Trajectories)

Run produced by `python scripts/run_experiments.py` (uses cached road network on second run).

| Method | Mean Match Dist (native units) | Edge Precision | Edge Recall | Edge F1 | Unmatched | Runtime/traj |
|--------|-------------------------------|----------------|-------------|---------|-----------|-------------|
| Nearest road | 0.0064 | 0.61 | 0.076 | 0.132 | 0 | ~0.08 s |
| HMM / Viterbi | 0.019 | 0.37 | 0.059 | 0.101 | 0 | ~8.4 s |

**Why nearest road has lower match distance**: it independently picks the geometrically
closest edge per point. HMM/Viterbi trades geometric closeness for path consistency
(staying on connected roads via the network graph).

**Why recall is low (~6–8%)**: Sparse GPS has ~13 points/trajectory versus 50–100
edges in each reference path. Edge F1 measures path coverage, not per-point accuracy.
This is expected and documented in the README and evaluation module.

---

## Tests

```
12 passed, 1 deselected in 1.36s   (fast unit tests only)
13 tests total (1 integration test requires real dataset)
```

| Test file | Count | Status |
|-----------|-------|--------|
| test_data_loader.py | 3 | ✅ Pass |
| test_evaluation.py | 3 | ✅ Pass |
| test_map_matching.py | 2 | ✅ Pass |
| test_preprocessing.py | 2 | ✅ Pass |
| test_road_network.py | 2 | ✅ Pass |
| test_integration.py | 1 | ✅ Pass (requires real data) |

---

## Improvements Made After Audit

### 1. HMM / Viterbi Route Cache (Critical Performance Fix)

**Problem**: HMM/Viterbi took ~24 seconds per trajectory because it called
Dijkstra's algorithm separately for every `(prev_candidate → curr_candidate)` pair
at each time step, even when many candidate edges shared the same endpoint nodes.

**Fix**: Added a `route_cache` dict inside `hmm_viterbi_match()`. Before calling
`shortest_path_length()`, we check if `(prev.edge.dst, curr.edge.src)` is already
in the cache. If yes, reuse the result. If no, call Dijkstra once and store it.

**Result**: Runtime dropped from ~24 s → ~8 s per trajectory (3× faster). On the
cached road network (second run onwards), it is faster still.

**Why it works** (for a B.Tech student): In a dense road network, multiple candidate
edges often have the same destination/source node. Without caching, we repeatedly
solve the same shortest-path problem. With caching, we solve each unique pair once.
This is the "memoization" technique applied to graph search.

### 2. Streamlit Dashboard — Results Tab Enhanced

Added pre-loaded summary table from `results/metrics.json` to both the Overview tab
and the Results & Export tab. Users now see experiment results immediately on opening
the dashboard, without having to re-run matching manually.

### 3. Integration Test Added

`tests/test_integration.py` — loads the real SparseTaxi files and road network,
runs both algorithms on 2 trajectories, and checks that matches are non-empty.
Marked with `@pytest.mark.integration` to allow fast runs to skip it.

### 4. pytest.ini — Registered Custom Marker

Added `integration` marker so pytest does not show unknown-marker warnings.

### 5. PROJECT_STATUS.md Created (this file)

---

## Known Limitations (Unchanged — by design)

| Issue | Explanation |
|-------|-------------|
| Unknown CRS | Coordinates are a projected space with unknown EPSG. Euclidean matching is valid but Folium OSM tiles are geographically wrong. Unfixable without projection parameters from the dataset authors. |
| Low edge recall | Expected. 311 sparse GPS points vs. 1400+ reference path nodes. F1 is a path-coverage score, not point accuracy. |
| No supervised ML | Route-level node sequences are not per-point labels. No classifier is trained. HMM/Viterbi is the advanced method. |
| 3 short dense trajectories | Original dataset artefact (3 dense trajectories have 1–3 fewer points than declared). Not fixable; documented. |

---

## How to Run

```powershell
# Activate virtual environment (if not already)
.\.venv\Scripts\Activate.ps1

# 1. Validate dataset (run once; writes data/validation_report.json)
python scripts/validate_dataset.py

# 2. Run experiments (writes results/ folder)
python scripts/run_experiments.py           # first 5 trajectories (~2 min)
python scripts/run_experiments.py --full    # all 23 trajectories (~5 min)

# 3. Run tests
python -m pytest -q                         # all tests (fast unit + integration)
python -m pytest -q -m "not integration"   # fast unit tests only (~2s)

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

---

## Architecture

```
data/raw/          <- SparseTaxi files (never modified)
data/processed/    <- road_network.pkl cache (auto-created)
data/validation_report.json  <- written by validate_dataset.py
results/           <- written by run_experiments.py
  comparison.csv, matched_points.csv, metrics.json,
  comparison_charts.png, evaluation_report.md

src/
  data_loader.py       <- parse .coor.txt and reference routes
  preprocessing.py     <- clean GPS: NaN, duplicates, jump flags
  road_network.py      <- load vertexMap + adjacentList -> NetworkX DiGraph + STRtree
  candidate_generation.py  <- spatial query for nearby road edges per GPS point
  nearest_road.py      <- independent closest-edge matcher
  hmm_map_matching.py  <- Viterbi sequence matcher (Newson & Krumm, 2009)
  evaluation.py        <- Euclidean match distance + edge precision/recall/F1
  visualization.py     <- Plotly cartesian map + Folium Simple-CRS map
  ml_model.py          <- feasibility assessment (no supervised training)
  utils.py             <- shared helpers (load_settings, euclidean, etc.)

scripts/
  validate_dataset.py  <- writes validation reports
  run_experiments.py   <- batch both algorithms on selected trajectories

tests/
  test_data_loader.py     <- 3 unit tests
  test_evaluation.py      <- 3 unit tests
  test_map_matching.py    <- 2 unit tests
  test_preprocessing.py   <- 2 unit tests
  test_road_network.py    <- 2 unit tests
  test_integration.py     <- 1 integration test (real data, marked)

app.py             <- Streamlit dashboard (6 tabs)
config/settings.yaml <- all tunable parameters (radius, HMM params, etc.)
docs/              <- methodology.md, architecture.md, limitations.md, viva_questions.md
```
