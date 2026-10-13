# GeoMotion AI

**An AI-based intelligent vehicle movement and map-matching system**  
Final-year B.Tech CSE (Artificial Intelligence and Machine Learning) prototype.

The program loads the selected SparseTaxi files, builds the supplied Beijing road graph, matches sparse GPS trajectories with a nearest-road baseline and an HMM/Viterbi decoder, evaluates what the files actually support, and shows the result in a Streamlit dashboard.

This is a full pipeline (data → graph → algorithms → metrics → UI), not a front-end-only demo.

## Problem

GPS points are noisy and, in this subset, sparse. Connecting them or snapping each one independently can pick the wrong road. The system uses road connectivity and sequence inference to reconstruct a more consistent route.

## Objectives

- Validate the real files (do not invent GPS points, road IDs or scores).
- Preprocess sparse trajectories transparently.
- Build a directed road graph from `vertexMap.txt` and `adjacentList.txt`.
- Implement nearest-road matching and HMM/Viterbi matching.
- Compare them with geometric distances and, when valid, reference path overlap.
- Demonstrate the work in Streamlit.

## Features

- Dataset validation report (`data/validation_report.json`)
- Configurable search radius and HMM parameters
- Interactive projected-coordinate maps (Plotly; Folium Simple CRS)
- CSV / GeoJSON export of a matched trajectory
- Pytest suite on small fixtures

## Technology stack

Python 3.11, Pandas/NumPy, NetworkX, Shapely, SciPy, scikit-learn (available; no supervised fit), Matplotlib, Plotly, Folium, Streamlit, PyYAML, Pytest.

OSMnx / GeoPandas are intentionally unused: the map is already local, and the CRS is not a known EPSG.

## Dataset

Source: [SparseTaxi](https://github.com/yangli-learning/SparseTaxi) selected files (copied into `data/raw`, originals left in Downloads).

See `data/README_DATASET.md` after you run the validator.

**Facts from file inspection (not marketing copy):**

- 23 sparse trajectories (311 points) aligned by **order_index** with 23 dense trajectories and 23 reference rows.
- Coordinates: shared projected space, overlap the graph; **not** treated as WGS84.
- Reference identifiers: node IDs in `vertexMap.txt` (upstream README said edge IDs; that does not match the file layout).

## Installation (Windows)

```powershell
cd C:\Users\Prachi\GeoMotionAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commands

```powershell
python scripts/validate_dataset.py
python -m pytest -q
python scripts/run_experiments.py
python scripts/run_experiments.py --full
streamlit run app.py
```

Default experiments use `demo.sample_mode` (first 5 trajectories). `--full` uses all 23.

## Algorithms (short)

- **Nearest road:** closest edge within the radius; project the point onto the segment.
- **HMM/Viterbi:** Gaussian emission on distance; exponential transition on `|network_length − GPS step|`; no invented shortcuts when the graph has no path.

Details: `docs/methodology.md`.

## Evaluation

Reports in `results/` are produced only by executing the experiment script. Geometric match distance ≠ ground-truth accuracy. Edge F1 uses reference **paths**, which are much denser than sparse GPS, so recall is typically low.

## Project layout

See `src/` for code, `tests/` for fixtures, `docs/` for viva notes, `config/settings.yaml` for parameters.

## Limitations and future work

`docs/limitations.md`. Main gaps: unknown CRS, placeholder timestamps, no point-wise ML labels.
