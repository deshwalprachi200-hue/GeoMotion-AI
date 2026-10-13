# Architecture

```
SPARSE GPS ──► data_loader ──► preprocessing ──► candidates
                                                     │
ROAD FILES ──► road_network (DiGraph + STRtree) ─────┤
                                                     ▼
                              nearest_road    hmm_map_matching
                                                     │
REFERENCE ──► evaluation (only after ID checks)      │
                                                     ▼
                         Streamlit app.py  /  results/
```

| Module | Role |
| --- | --- |
| `src/data_loader.py` | Parse `.coor.txt` and reference rows; keep `order_index` |
| `src/preprocessing.py` | Finite coords, duplicates, jump flags; do not fake timestamps |
| `src/road_network.py` | Nodes, directed edges, vertex-to-vertex geometry, OSM attributes |
| `src/candidate_generation.py` | Radius query + point-to-segment distance |
| `src/nearest_road.py` | Baseline |
| `src/hmm_map_matching.py` | Custom HMM + Viterbi (no hmmlearn) |
| `src/ml_model.py` | Feasibility report only |
| `src/evaluation.py` | Distances and optional edge F1 |
| `src/visualization.py` | Plotly cartesian map + Folium Simple CRS |
| `app.py` | Six dashboard sections |
| `scripts/validate_dataset.py` | Writes `data/validation_report.*` |
| `scripts/run_experiments.py` | Writes `results/` |

Algorithms do not import Streamlit, so tests can run without the UI.

OSMnx is **not** used: the road network is already in SparseTaxi files. Forcing OSMnx would download a different map. GeoPandas/PyProj are omitted because there is no known EPSG code to reproject; Shapely handles geometry.

Caching: `data/processed/road_network.pkl` plus Streamlit `@st.cache_resource`.
