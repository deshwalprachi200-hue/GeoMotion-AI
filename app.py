"""GeoMotion AI — Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_project_datasets
from src.evaluation import eval_to_row, evaluate_result
from src.hmm_map_matching import HmmParameters, hmm_viterbi_match
from src.ml_model import assess_supervised_feasibility
from src.nearest_road import nearest_road_match
from src.preprocessing import preprocess_all
from src.road_network import load_or_build_network
from src.utils import load_settings
from src.visualization import cartesian_network_figure, folium_map, match_export_rows

st.set_page_config(page_title="GeoMotion AI", layout="wide", page_icon="🗺️")


st.markdown(
    """
    <style>
    /* ===== GeoMotion AI: Global Theme ===== */

    .stApp {
        background-color: #F4F1EA;
        color: #1D2939;
    }

    /* Main content area */
    [data-testid="stAppViewContainer"] {
        background-color: #F4F1EA;
    }

    [data-testid="stMain"] {
        background-color: #F4F1EA;
    }

    [data-testid="stHeader"] {
        background-color: rgba(244, 241, 234, 0.95);
    }

    /* All regular text */
    .stApp,
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp li,
    .stApp small,
    .stApp [data-testid="stCaptionContainer"] {
        color: #263445;
    }

    /* Headings */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #1D3557 !important;
        font-family: "Segoe UI", sans-serif;
        font-weight: 650;
    }

    /* Main title */
    .stApp h1 {
        font-size: 2.4rem;
        letter-spacing: -0.5px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #D6DCE5;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #344054 !important;
        border-radius: 8px 8px 0 0;
        padding: 10px 14px;
    }

    .stTabs [data-baseweb="tab"] p {
        color: #344054 !important;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 3px solid #167D8D;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
        color: #1D3557 !important;
        font-weight: 700;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E5EC;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 7px rgba(29, 53, 87, 0.06);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #475467 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #1D3557 !important;
        font-weight: 700;
    }

    /* Cards and bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #DDE3EA;
        border-radius: 12px;
    }

    /* Dataframes and tables */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background-color: #FFFFFF;
        border-radius: 8px;
    }

    /* Inputs, dropdowns and multiselects */
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="textarea"] > div {
        background-color: #FFFFFF;
        border-color: #C9D2DF;
    }

    .stApp input,
    .stApp textarea {
        color: #1D2939 !important;
        background-color: #FFFFFF !important;
    }

    .stApp [data-baseweb="select"] *,
    .stApp [data-baseweb="popover"] * {
        color: #1D2939;
    }

    /* Buttons */
    .stApp .stButton > button,
    .stApp .stDownloadButton > button {
        background-color: #167D8D;
        color: #FFFFFF !important;
        border: 1px solid #167D8D;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.45rem 1rem;
    }

    .stApp .stButton > button p,
    .stApp .stDownloadButton > button p {
        color: #FFFFFF !important;
    }

    .stApp .stButton > button:hover,
    .stApp .stDownloadButton > button:hover {
        background-color: #116675;
        border-color: #116675;
    }

    /* Radio buttons and checkboxes */
    .stApp [data-testid="stRadio"] label,
    .stApp [data-testid="stCheckbox"] label {
        color: #263445 !important;
    }

    /* Slider labels */
    .stApp [data-testid="stSlider"] {
        color: #263445;
    }

    /* Expanders */
    .stApp [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #DDE3EA;
        border-radius: 10px;
    }

    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary p {
        color: #1D3557 !important;
    }

    /* JSON blocks and code */
    .stApp [data-testid="stJson"],
    .stApp [data-testid="stCode"],
    .stApp pre {
        background-color: #FFFFFF;
        color: #263445;
        border-radius: 8px;
    }

    /* Warning box */
    .warn {
        background-color: #FFF3CD;
        color: #664D03;
        padding: 0.85rem 1rem;
        border: 1px solid #FFECB5;
        border-radius: 8px;
    }

    .warn p,
    .warn strong {
        color: #664D03 !important;
    }

    /* Success, info and error messages */
    [data-testid="stAlert"] p {
        color: #263445 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E8EDEB;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #263445;
    }

    /* Links */
    .stApp a {
        color: #116675;
    }

    /* Captions */
    .stApp [data-testid="stCaptionContainer"] p {
        color: #596579 !important;
    }

    /* Keep maps visible */
    [data-testid="stIFrame"],
    iframe {
        background-color: #FFFFFF;
        border-radius: 8px;
    }

    /* Mobile / narrow window */
    @media (max-width: 768px) {
        .stApp h1 {
            font-size: 1.8rem;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)



@st.cache_resource(show_spinner="Loading road network…")
def get_network():
    return load_or_build_network()


@st.cache_data(show_spinner="Loading trajectories…")
def get_bundle():
    settings = load_settings()
    bundle = load_project_datasets(settings)
    sparse = preprocess_all(bundle["sparse"], settings)
    dense = preprocess_all(bundle["dense"], settings)
    return settings, bundle, sparse, dense


def main() -> None:
    try:
        settings, bundle, sparse, dense = get_bundle()
        network = get_network()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load data or road network. {exc}")
        st.info("Confirm `data/raw` exists and run `python scripts/validate_dataset.py`.")
        return

    refs = {r.order_index: r for r in bundle["references"]}
    validation_path = ROOT / "data" / "validation_report.json"
    validation = {}
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))

    st.title("GeoMotion AI")
    st.caption(
        "Intelligent vehicle movement and map-matching — SparseTaxi Beijing subset, "
        "processed offline in a projected coordinate space."
    )

    tabs = st.tabs(
        [
            "Overview",
            "Dataset Explorer",
            "Road Network",
            "Map Matching",
            "Algorithm Comparison",
            "Results and Export",
        ]
    )

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sparse trajectories", len(sparse))
        c2.metric("Sparse GPS points", sum(len(t.points) for t in sparse))
        c3.metric("Road nodes", network.diagnostics["n_nodes"])
        c4.metric("Directed edges", network.diagnostics["n_edges"])
        st.subheader("Objective")
        st.write(
            "Take noisy or sparse GPS trajectories, find likely roads, reconstruct "
            "the route on the supplied road network, and compare nearest-road matching "
            "with HMM/Viterbi sequence matching."
        )
        st.subheader("Available algorithms")
        st.write("- **Nearest road** — independent closest edge per observation")
        st.write("- **HMM / Viterbi** — sequence model with distance emissions and network transitions")
        st.write("- **Supervised ML** — not trained (route-level labels only)")
        if validation.get("issues"):
            st.markdown('<div class="warn">', unsafe_allow_html=True)
            st.write(f"Validation warnings/issues: **{len(validation['issues'])}**. See Dataset Explorer.")
            st.markdown("</div>", unsafe_allow_html=True)
        st.json(
            {
                "coordinate_system": settings["coordinate_system"]["kind"],
                "gps_overlaps_network": validation.get("gps_overlaps_network"),
                "alignment_key": "order_index",
            }
        )
        # Show saved experiment summary if available
        metrics_path = ROOT / "results" / "metrics.json"
        if metrics_path.exists():
            st.subheader("Last experiment results")
            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            nr_m = saved_metrics.get("nearest_road", {})
            hmm_m = saved_metrics.get("hmm_viterbi", {})
            summary_rows = [
                {
                    "Method": "Nearest road",
                    "Trajectories": nr_m.get("n_trajectories"),
                    "Mean match dist (native)": round(nr_m.get("mean_of_mean_match_distance") or 0, 5),
                    "Edge precision": round(nr_m.get("mean_edge_precision") or 0, 3),
                    "Edge recall": round(nr_m.get("mean_edge_recall") or 0, 3),
                    "Edge F1": round(nr_m.get("mean_edge_f1") or 0, 3),
                    "Unmatched": nr_m.get("total_unmatched"),
                    "Runtime/traj (s)": round(nr_m.get("mean_runtime_sec") or 0, 3),
                },
                {
                    "Method": "HMM / Viterbi",
                    "Trajectories": hmm_m.get("n_trajectories"),
                    "Mean match dist (native)": round(hmm_m.get("mean_of_mean_match_distance") or 0, 5),
                    "Edge precision": round(hmm_m.get("mean_edge_precision") or 0, 3),
                    "Edge recall": round(hmm_m.get("mean_edge_recall") or 0, 3),
                    "Edge F1": round(hmm_m.get("mean_edge_f1") or 0, 3),
                    "Unmatched": hmm_m.get("total_unmatched"),
                    "Runtime/traj (s)": round(hmm_m.get("mean_runtime_sec") or 0, 3),
                },
            ]
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
            st.caption(
                f"Sample mode={saved_metrics.get('sample_mode')} — "
                f"trajectories {saved_metrics.get('trajectories_evaluated')}. "
                "Run `python scripts/run_experiments.py --full` for all 23."
            )
            chart_path = ROOT / "results" / "comparison_charts.png"
            if chart_path.exists():
                st.image(str(chart_path), caption="Comparison charts (from last experiment run)")

    with tabs[1]:
        options = [f"{t.order_index} (file_id={t.file_trajectory_id}, n={len(t.points)})" for t in sparse]
        choice = st.selectbox("Sparse trajectory (aligned by order_index)", options, key="exp")
        tid = int(choice.split()[0])
        sparse_t = next(t for t in sparse if t.order_index == tid)
        dense_t = next((t for t in dense if t.order_index == tid), None)
        st.write("Sparse file IDs are placeholders; this index matches dense GPS and the reference row.")
        left, right = st.columns(2)
        with left:
            st.markdown("**Cleaned sparse points**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "i": p.sequence_index,
                            "timestamp": p.timestamp,
                            "x": p.x,
                            "y": p.y,
                            "dt": p.dt,
                            "displacement": p.displacement,
                            "jump": p.implausible_jump,
                        }
                        for p in sparse_t.points
                    ]
                ),
                hide_index=True,
            )
            st.caption("Timestamps are treated as placeholders (regular 60-unit steps), not clock time.")
        with right:
            st.markdown("**Dense GPS (manual-matched file)**")
            if dense_t:
                st.dataframe(
                    pd.DataFrame(
                        [{"i": p.sequence_index, "timestamp": p.timestamp, "x": p.x, "y": p.y} for p in dense_t.points[:50]]
                    ),
                    hide_index=True,
                )
                st.caption("Showing up to 50 dense points.")
            st.markdown("**Trajectory warnings**")
            st.write(sparse_t.warnings + (dense_t.warnings if dense_t else []))
            st.write("Removed records:", sparse_t.removed)
        if validation.get("issues"):
            st.markdown("**Dataset validation issues**")
            st.dataframe(pd.DataFrame(validation["issues"]))

    with tabs[2]:
        st.write(network.diagnostics["geometry_source"])
        st.write(
            "OSM polylines were not used as invented geometry; highway type/name/speed "
            "are joined when way node IDs match the graph."
        )
        stats = network.diagnostics
        st.json(
            {
                "extent": network.extent(),
                "weakly_connected_components": stats["weakly_connected_components"],
                "strongly_connected_components": stats["strongly_connected_components"],
                "osm_join": stats.get("osm_join"),
                "adjacentList_first_line_note": stats["declared_vs_actual_note"],
            }
        )
        st.plotly_chart(cartesian_network_figure(network, max_edges=3000), use_container_width=True)

    with tabs[3]:
        options = [f"{t.order_index} (n={len(t.points)})" for t in sparse]
        choice = st.selectbox("Trajectory", options, key="match_t")
        tid = int(choice.split()[0])
        traj = next(t for t in sparse if t.order_index == tid)
        method = st.radio("Algorithm", ["nearest_road", "hmm_viterbi"], horizontal=True)
        radius = st.slider("Search radius (native units)", 0.1, 3.0, float(settings["matching"]["search_radius"]), 0.1)
        max_c = st.slider("Max candidates", 3, 15, int(settings["matching"]["max_candidates"]))
        sigma = st.slider("HMM GPS sigma", 0.05, 1.0, float(settings["matching"]["gps_sigma"]), 0.05)
        beta = st.slider("HMM transition beta", 0.2, 8.0, float(settings["matching"]["transition_beta"]), 0.2)
        if st.button("Run map matching", type="primary"):
            with st.spinner("Matching…"):
                if method == "nearest_road":
                    result = nearest_road_match(traj, network, search_radius=radius, max_candidates=max_c)
                else:
                    result = hmm_viterbi_match(
                        traj,
                        network,
                        HmmParameters(
                            search_radius=radius,
                            max_candidates=max_c,
                            gps_sigma=sigma,
                            transition_beta=beta,
                        ),
                    )
            st.session_state["last_result"] = result
            st.session_state["last_traj"] = traj
        result = st.session_state.get("last_result")
        traj_show = st.session_state.get("last_traj", traj)
        if result is None:
            st.info("Run matching to draw GPS, matched positions and unresolved points.")
        else:
            st.write(
                f"{result.method}: matched {result.matched_count}/{len(result.records)} "
                f"in {result.runtime_sec:.2f}s"
            )
            st.write("Notes:", result.notes)
            st.plotly_chart(
                cartesian_network_figure(network, traj_show, result, max_edges=2500),
                use_container_width=True,
            )
            with st.expander("Folium view (Simple CRS, no OSM tiles)"):
                st_folium(folium_map(network, traj_show, result, max_edges=800), height=480, width=None)
            st.dataframe(pd.DataFrame(match_export_rows(result)), hide_index=True)

    with tabs[4]:
        st.write(
            "Comparison uses the same trajectory, radius and candidate cap for both methods. "
            "Edge precision/recall need a verified reference row; they are not geometric distance."
        )
        options = [f"{t.order_index} (n={len(t.points)})" for t in sparse]
        choice = st.selectbox("Trajectory", options, key="cmp_t")
        tid = int(choice.split()[0])
        traj = next(t for t in sparse if t.order_index == tid)
        if st.button("Compare nearest-road vs HMM"):
            with st.spinner("Running both algorithms…"):
                radius = float(settings["matching"]["search_radius"])
                max_c = int(settings["matching"]["max_candidates"])
                base = nearest_road_match(traj, network, search_radius=radius, max_candidates=max_c)
                hmm = hmm_viterbi_match(traj, network, HmmParameters(
                    search_radius=radius,
                    max_candidates=max_c,
                    gps_sigma=float(settings["matching"]["gps_sigma"]),
                    transition_beta=float(settings["matching"]["transition_beta"]),
                ))
                route = refs.get(tid)
                evs = [evaluate_result(base, route, network), evaluate_result(hmm, route, network)]
                st.session_state["cmp_table"] = pd.DataFrame([eval_to_row(e) for e in evs])
                st.session_state["cmp_notes"] = evs[0].notes + evs[1].notes
        if "cmp_table" in st.session_state:
            st.dataframe(st.session_state["cmp_table"], hide_index=True)
            st.caption("Distance columns are native projected units. F1 uses reference path edges.")
            st.write(st.session_state["cmp_notes"])
        saved = ROOT / "results" / "comparison.csv"
        if saved.exists():
            st.markdown("**Saved experiment file** (`results/comparison.csv`)")
            st.dataframe(pd.read_csv(saved), hide_index=True)

    with tabs[5]:
        st.write("Exports use matching output from the Map Matching tab or saved experiment files.")

        # Show saved experiment summary
        metrics_path = ROOT / "results" / "metrics.json"
        report_path = ROOT / "results" / "evaluation_report.md"
        chart_path = ROOT / "results" / "comparison_charts.png"
        if metrics_path.exists():
            st.subheader("Saved experiment results")
            st.info(
                "These results were written by `python scripts/run_experiments.py`. "
                "Re-run the script to refresh them."
            )
            saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            nr_m = saved_metrics.get("nearest_road", {})
            hmm_m = saved_metrics.get("hmm_viterbi", {})
            summary_rows = [
                {
                    "Method": "Nearest road",
                    "N trajectories": nr_m.get("n_trajectories"),
                    "Mean match dist": round(nr_m.get("mean_of_mean_match_distance") or 0, 6),
                    "Edge precision": round(nr_m.get("mean_edge_precision") or 0, 4),
                    "Edge recall": round(nr_m.get("mean_edge_recall") or 0, 4),
                    "Edge F1": round(nr_m.get("mean_edge_f1") or 0, 4),
                    "Total unmatched": nr_m.get("total_unmatched"),
                    "Mean runtime (s)": round(nr_m.get("mean_runtime_sec") or 0, 4),
                },
                {
                    "Method": "HMM / Viterbi",
                    "N trajectories": hmm_m.get("n_trajectories"),
                    "Mean match dist": round(hmm_m.get("mean_of_mean_match_distance") or 0, 6),
                    "Edge precision": round(hmm_m.get("mean_edge_precision") or 0, 4),
                    "Edge recall": round(hmm_m.get("mean_edge_recall") or 0, 4),
                    "Edge F1": round(hmm_m.get("mean_edge_f1") or 0, 4),
                    "Total unmatched": hmm_m.get("total_unmatched"),
                    "Mean runtime (s)": round(hmm_m.get("mean_runtime_sec") or 0, 4),
                },
            ]
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
            st.caption(
                "'Mean match dist' = Euclidean GPS-to-road distance in native projected units. "
                "Edge F1 uses reference path edges, not per-point labels — "
                "recall is expected to be low for sparse GPS."
            )
            if chart_path.exists():
                st.image(str(chart_path), caption="Match distance and edge F1 comparison (last experiment run)")
            if report_path.exists():
                st.markdown(report_path.read_text(encoding="utf-8"))
        else:
            st.info("No saved experiment results yet. Run: `python scripts/run_experiments.py`")

        st.divider()
        st.subheader("Download from this session")
        result = st.session_state.get("last_result")
        if result is not None:
            csv_bytes = pd.DataFrame(match_export_rows(result)).to_csv(index=False).encode("utf-8")
            st.download_button("Download last match CSV", csv_bytes, "matched_trajectory.csv", "text/csv")
            geo = {
                "type": "FeatureCollection",
                "crs_note": "native projected coordinates, not EPSG:4326",
                "features": [],
            }
            for rec in result.records:
                geo["features"].append(
                    {
                        "type": "Feature",
                        "properties": {
                            "matched": rec.matched,
                            "edge_id": rec.edge_id,
                            "distance": rec.distance,
                        },
                        "geometry": {"type": "Point", "coordinates": [rec.gps_x, rec.gps_y]},
                    }
                )
            st.download_button(
                "Download last match GeoJSON (native coords)",
                json.dumps(geo, indent=2),
                "matched_trajectory.geojson",
                "application/geo+json",
            )
        else:
            st.info("Run map matching in the 'Map Matching' tab first to enable session download.")

        st.divider()
        st.subheader("Download saved files")
        for name in ("metrics.json", "evaluation_report.md", "comparison.csv", "matched_points.csv"):
            path = ROOT / "results" / name
            if path.exists():
                st.download_button(
                    f"Download {name}",
                    path.read_bytes(),
                    name,
                    key=f"dl_{name}",
                )

        st.divider()
        st.subheader("Reproducibility")
        st.json(
            {
                "dataset": "SparseTaxi selected subset in data/raw",
                "settings": settings["matching"],
                "python_entrypoints": [
                    "python scripts/validate_dataset.py",
                    "python scripts/run_experiments.py",
                    "python scripts/run_experiments.py --full",
                    "streamlit run app.py",
                ],
                "supervised_training_performed": assess_supervised_feasibility(311, 23, True)["supervised_training_performed"],
            }
        )


main()
