"""Supervised-learning feasibility check — no fabricated labels."""

from __future__ import annotations


def assess_supervised_feasibility(
    n_sparse_points: int,
    n_reference_routes: int,
    reference_is_node_path: bool,
) -> dict:
    """The selected SparseTaxi files give route-level node sequences, not
    per-observation road labels. Training a point classifier would require
    inventing a label for every GPS sample (for example nearest reference
    edge). That heuristic is not ground truth and would leak route geometry
    into the labels.

    HMM/Viterbi remains the advanced method for this submission.
    """
    reasons = [
        "Reference files contain one node sequence per trajectory, not a label per GPS point.",
        "Sparse observations (311 points) do not align 1:1 with dense/reference vertices.",
        "Randomly splitting points from the same trip would leak trajectory identity.",
    ]
    if reference_is_node_path:
        reasons.append(
            "Identifiers match vertexMap node IDs; they are a path, not candidate-class labels."
        )
    return {
        "supervised_training_performed": False,
        "primary_advanced_method": "hmm_viterbi",
        "n_sparse_points": n_sparse_points,
        "n_reference_routes": n_reference_routes,
        "reasons": reasons,
        "if_labels_become_available": [
            "Split by trajectory order_index, never by shuffling adjacent points.",
            "Keep reference path out of feature columns.",
            "Report candidate-ranking metrics separately from HMM sequence metrics.",
        ],
    }
