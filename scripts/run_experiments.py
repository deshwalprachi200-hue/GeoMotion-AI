"""Run baseline + HMM on eligible sparse trajectories and write results/."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_project_datasets
from src.evaluation import eval_to_row, evaluate_result
from src.hmm_map_matching import hmm_viterbi_match, parameters_from_settings
from src.ml_model import assess_supervised_feasibility
from src.nearest_road import nearest_road_match
from src.preprocessing import preprocess_all
from src.road_network import load_or_build_network
from src.utils import ensure_dir, load_settings, project_root, write_json
from src.visualization import match_export_rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main(sample_mode: bool | None = None) -> None:
    settings = load_settings()
    if sample_mode is None:
        sample_mode = bool(settings["demo"]["sample_mode"])
    bundle = load_project_datasets(settings)
    network = load_or_build_network(settings)
    sparse = preprocess_all(bundle["sparse"], settings)
    refs = {route.order_index: route for route in bundle["references"]}
    params = parameters_from_settings(settings)

    limit = len(sparse)
    if sample_mode:
        limit = min(len(sparse), int(settings["demo"]["sample_trajectory_count"]))
    selected = sparse[:limit]

    rows = []
    match_rows = []
    evals = []
    failed = []
    for traj in selected:
        if len(traj.points) < 2:
            failed.append({"order_index": traj.order_index, "reason": "fewer_than_2_points"})
            continue
        route = refs.get(traj.order_index)
        try:
            baseline = nearest_road_match(
                traj,
                network,
                search_radius=params.search_radius,
                max_candidates=params.max_candidates,
            )
            hmm = hmm_viterbi_match(traj, network, params)
        except Exception as exc:  # noqa: BLE001 — record and continue
            failed.append({"order_index": traj.order_index, "reason": str(exc)})
            continue
        for result in (baseline, hmm):
            ev = evaluate_result(result, route, network)
            evals.append(ev)
            rows.append(eval_to_row(ev))
            match_rows.extend(match_export_rows(result))

    out = ensure_dir(project_root() / "results")
    comparison_path = out / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    matches_path = out / "matched_points.csv"
    with matches_path.open("w", newline="", encoding="utf-8") as handle:
        if match_rows:
            writer = csv.DictWriter(handle, fieldnames=list(match_rows[0].keys()))
            writer.writeheader()
            writer.writerows(match_rows)

    def summarize(method: str) -> dict:
        subset = [row for row in rows if row["method"] == method]
        dist = [row["mean_match_distance"] for row in subset if row["mean_match_distance"] is not None]
        prec = [row["edge_precision"] for row in subset if row["edge_precision"] is not None]
        rec = [row["edge_recall"] for row in subset if row["edge_recall"] is not None]
        f1 = [row["edge_f1"] for row in subset if row["edge_f1"] is not None]
        return {
            "n_trajectories": len(subset),
            "mean_of_mean_match_distance": _mean(dist),
            "mean_edge_precision": _mean(prec),
            "mean_edge_recall": _mean(rec),
            "mean_edge_f1": _mean(f1),
            "mean_runtime_sec": _mean([row["runtime_sec"] for row in subset]),
            "total_unmatched": sum(row["unmatched_count"] for row in subset),
        }

    metrics = {
        "sample_mode": sample_mode,
        "trajectories_evaluated": [t.order_index for t in selected],
        "parameters": params.__dict__,
        "nearest_road": summarize("nearest_road"),
        "hmm_viterbi": summarize("hmm_viterbi"),
        "failed_or_excluded": failed,
        "supervised_learning": assess_supervised_feasibility(
            sum(len(t.points) for t in sparse),
            len(refs),
            True,
        ),
        "metric_definitions": {
            "mean_match_distance": "Euclidean GPS-to-matched-road distance in native projected units (not metres unless 1 unit happens to be 1 m).",
            "edge_precision_recall": "Set overlap between predicted observation edges and reference path edges. Not a point-accuracy score.",
        },
    }
    write_json(out / "metrics.json", metrics)

    if rows:
        methods = ["nearest_road", "hmm_viterbi"]
        labels = ["Nearest road", "HMM / Viterbi"]
        dist_vals = [summarize(m)["mean_of_mean_match_distance"] or 0 for m in methods]
        f1_vals = [summarize(m)["mean_edge_f1"] or 0 for m in methods]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].bar(labels, dist_vals, color=["#E85D04", "#2A9D8F"])
        axes[0].set_title("Mean GPS-to-road distance")
        axes[0].set_ylabel("native units")
        axes[1].bar(labels, f1_vals, color=["#E85D04", "#2A9D8F"])
        axes[1].set_title("Mean edge F1 vs reference path")
        axes[1].set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out / "comparison_charts.png", dpi=140)
        plt.close(fig)

    report = out / "evaluation_report.md"
    nr, hmm = summarize("nearest_road"), summarize("hmm_viterbi")
    report.write_text(
        f"""# Evaluation report

Generated by `python scripts/run_experiments.py`.

- Trajectories evaluated: `{metrics['trajectories_evaluated']}` (sample_mode={sample_mode})
- Search radius: {params.search_radius} native units
- HMM sigma={params.gps_sigma}, beta={params.transition_beta}

## Geometric distance (no ground-truth needed)

| Method | Mean of per-traj mean match distance | Unmatched observations | Mean runtime (s) |
| --- | --- | --- | --- |
| Nearest road | {nr['mean_of_mean_match_distance']} | {nr['total_unmatched']} | {nr['mean_runtime_sec']} |
| HMM / Viterbi | {hmm['mean_of_mean_match_distance']} | {hmm['total_unmatched']} | {hmm['mean_runtime_sec']} |

## Reference path edge overlap (use with care)

Reference rows are node sequences, not per-point labels. Sparse GPS has far fewer
points than each reference path, so **recall is expected to be low**. These scores
are not “accuracy %” for a paper leaderboard.

| Method | Mean precision | Mean recall | Mean F1 |
| --- | --- | --- | --- |
| Nearest road | {nr['mean_edge_precision']} | {nr['mean_edge_recall']} | {nr['mean_edge_f1']} |
| HMM / Viterbi | {hmm['mean_edge_precision']} | {hmm['mean_edge_recall']} | {hmm['mean_edge_f1']} |

Failed/excluded: {json.dumps(failed)}

Supervised ML was **not** trained. See `src/ml_model.py`.
""",
        encoding="utf-8",
    )
    failed_path = out / "excluded_trajectories.json"
    write_json(failed_path, failed)
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    sample = None
    if "--full" in sys.argv:
        sample = False
    main(sample_mode=sample)
