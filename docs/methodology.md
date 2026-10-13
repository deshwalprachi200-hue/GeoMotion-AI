# Methodology

This document matches the code in `src/`. It is written for a B.Tech CSE (AI/ML) viva.

## GPS noise and sparsity

A GPS receiver reports an estimated position, not the exact lane of the vehicle. Error, urban canyons and sparse sampling mean that drawing straight lines between observations, or snapping each point to the nearest road, can pick the wrong road when two roads run side by side.

GeoMotion AI therefore:

1. Loads the **sparse** trajectories as the noisy input.
2. Builds the **road graph** from `vertexMap.txt` + `adjacentList.txt`.
3. Finds **candidate roads** near each observation.
4. Compares two algorithms: **nearest road** (independent) and **HMM + Viterbi** (sequence-aware).
5. Scores results with geometric distances and, where a reference node path exists, edge-set overlap.

## Coordinates

The files do **not** contain an EPSG code. GPS `x,y`, graph vertices and OSM way vertices occupy the same numeric range and overlap in space. The map README calls this a projected space based on longitude/latitude. Distances in the code are **Euclidean in those units**. They are not claimed to be metres, and they are not haversine kilometres.

Timestamps in this subset look like placeholders (dense: 0,1,2,…; sparse: 60,120,180,…). Speed in km/h is **not** treated as a real physical speed.

## Candidate generation

For each GPS point the code queries a Shapely `STRtree` of edge segments within `search_radius`. Each hit is scored by perpendicular (point-to-segment) distance. Heading difference is stored when two successive points define a heading. Observations with zero candidates are left unmatched; the matcher does not invent a road.

## Nearest-road baseline

For every observation, take the candidate with the smallest distance and project the GPS point onto that segment. This is easy to explain and is the comparison baseline. It can flicker between parallel roads because it ignores the rest of the trip.

## Hidden Markov Model

- **Hidden states:** candidate road positions (an observation may have several nearby edges).
- **Emissions:** GPS points.
- **Emission log-probability:** Gaussian in point-to-road distance with configurable `gps_sigma` (Newson & Krumm style).
- **Transition log-probability:** `−|network_distance − observed_displacement| / beta`, where network distance is a shortest-path length on the directed graph (with a distance cutoff so the laptop does not search the whole city).
- If no path exists between two candidates, that transition is **not** invented. Viterbi uses a documented emission-only fallback and counts it as unresolved.

Viterbi finds the most likely **sequence** of candidates. That is the advanced algorithm in this project.

## Why this is AI/ML without a trained classifier

HMM sequence decoding is a standard probabilistic inference method. A Random Forest would need a class label for every GPS point. The reference file is a **node path for the whole trip**, not those labels. Inventing labels (for example “nearest reference edge”) would be circular and leak the answer into training. `src/ml_model.py` records that supervised training was **not** performed.

## Evaluation

- **Match distance / match rate:** how close the chosen road is to the GPS point. Does not prove the road is the true road.
- **Edge precision / recall / F1:** predicted `(src,dst)` pairs versus edges implied by consecutive reference **node IDs**. Sparse GPS has far fewer points than a dense node path, so recall is usually low even when matching is reasonable.
- Reference IDs are treated as **vertex IDs** after validation against `vertexMap.txt`, not as `adjacentList` row numbers.

All numbers in `results/` come from running `scripts/run_experiments.py`. They are not hardcoded.
