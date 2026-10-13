"""Interactive maps. Native coordinates are projected, not WGS84 lat/lon."""

from __future__ import annotations

import folium
import plotly.graph_objects as go

from src.nearest_road import MatchResult
from src.preprocessing import PreprocessedTrajectory
from src.road_network import RoadNetwork

CRS_WARNING = (
    "Coordinates are a custom projected space from SparseTaxi, not WGS84. "
    "OpenStreetMap tiles will not show Beijing under these numbers."
)


def cartesian_network_figure(
    network: RoadNetwork,
    trajectory: PreprocessedTrajectory | None = None,
    result: MatchResult | None = None,
    max_edges: int = 4000,
) -> go.Figure:
    edges = network.edges
    step = max(1, len(edges) // max_edges)
    sample = edges[::step]
    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    for edge in sample:
        x_lines.extend([edge.x1, edge.x2, None])
        y_lines.extend([edge.y1, edge.y2, None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line={"width": 1, "color": "#4C6B8A"},
            name="Road network (sampled)",
            hoverinfo="skip",
        )
    )
    if trajectory is not None and trajectory.points:
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in trajectory.points],
                y=[p.y for p in trajectory.points],
                mode="markers+lines",
                marker={"size": 8, "color": "#E85D04"},
                line={"color": "#E85D04", "width": 2},
                name="GPS observations",
            )
        )
    if result is not None:
        mx = [r.matched_x for r in result.records if r.matched]
        my = [r.matched_y for r in result.records if r.matched]
        if mx:
            fig.add_trace(
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="markers+lines",
                    marker={"size": 8, "color": "#2A9D8F", "symbol": "x"},
                    line={"color": "#2A9D8F", "width": 2},
                    name="Matched positions",
                )
            )
        ux = [r.gps_x for r in result.records if not r.matched]
        uy = [r.gps_y for r in result.records if not r.matched]
        if ux:
            fig.add_trace(
                go.Scatter(
                    x=ux,
                    y=uy,
                    mode="markers",
                    marker={"size": 10, "color": "#9B2226", "symbol": "diamond"},
                    name="Unmatched GPS",
                )
            )
    fig.update_layout(
        template="plotly_white",
        title="Projected coordinate map (not a geographic basemap)",
        xaxis_title="x (native units)",
        yaxis_title="y (native units)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h"},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        height=560,
    )
    fig.add_annotation(
        text=CRS_WARNING,
        xref="paper",
        yref="paper",
        x=0,
        y=1.12,
        showarrow=False,
        font={"size": 11, "color": "#6C584C"},
        align="left",
    )
    return fig


def folium_map(
    network: RoadNetwork,
    trajectory: PreprocessedTrajectory | None = None,
    result: MatchResult | None = None,
    max_edges: int = 1500,
) -> folium.Map:
    """Folium map using Simple CRS so we do not pretend the values are lat/lon."""
    extent = network.extent()
    cx = 0.5 * (extent["min_x"] + extent["max_x"])
    cy = 0.5 * (extent["min_y"] + extent["max_y"])
    fmap = folium.Map(location=[cy, cx], tiles=None, crs="Simple", zoom_start=2)
    step = max(1, len(network.edges) // max_edges)
    for edge in network.edges[::step]:
        folium.PolyLine(
            locations=[(edge.y1, edge.x1), (edge.y2, edge.x2)],
            color="#4C6B8A",
            weight=1,
            opacity=0.5,
        ).add_to(fmap)
    if trajectory is not None:
        gps = [(p.y, p.x) for p in trajectory.points]
        if gps:
            folium.PolyLine(gps, color="#E85D04", weight=3, tooltip="GPS").add_to(fmap)
            for point in trajectory.points:
                folium.CircleMarker(
                    location=(point.y, point.x),
                    radius=4,
                    color="#E85D04",
                    fill=True,
                    tooltip=f"GPS t={point.timestamp}",
                ).add_to(fmap)
    if result is not None:
        matched = [(r.matched_y, r.matched_x) for r in result.records if r.matched and r.matched_x is not None]
        if matched:
            folium.PolyLine(matched, color="#2A9D8F", weight=3, tooltip="Matched").add_to(fmap)
    fmap.get_root().html.add_child(
        folium.Element(f"<div style='padding:8px;background:#fff3cd;font-size:13px'>{CRS_WARNING}</div>")
    )
    return fmap


def match_export_rows(result: MatchResult) -> list[dict]:
    rows = []
    for rec in result.records:
        rows.append(
            {
                "observation_index": rec.observation_index,
                "method": result.method,
                "gps_x": rec.gps_x,
                "gps_y": rec.gps_y,
                "matched": rec.matched,
                "edge_id": rec.edge_id,
                "src": rec.src,
                "dst": rec.dst,
                "matched_x": rec.matched_x,
                "matched_y": rec.matched_y,
                "distance": rec.distance,
                "unresolved_reason": rec.unresolved_reason,
            }
        )
    return rows
