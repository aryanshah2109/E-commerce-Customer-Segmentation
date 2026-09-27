"""Plotly chart constructors shared by dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from streamlit_app.utils.helpers import CLUSTER_COLORS


def _apply_light_theme(figure: go.Figure) -> go.Figure:
    """Apply the dashboard's light palette to a Plotly figure."""
    figure.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#172033", "family": "DM Sans, sans-serif"},
        title_font={"color": "#172033", "family": "Space Grotesk, sans-serif"},
        legend={"font": {"color": "#172033"}},
        margin={"l": 48, "r": 24, "t": 56, "b": 48},
    )
    figure.update_xaxes(
        color="#53627a",
        gridcolor="#e3e8ef",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
    )
    figure.update_yaxes(
        color="#53627a",
        gridcolor="#e3e8ef",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
    )
    return figure


def line_chart(
    data_frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_title: str,
) -> go.Figure:
    """Build a consistent line chart."""
    figure = px.line(data_frame, x=x, y=y, title=title, markers=True)
    figure.update_traces(line_color="#2F6BFF", hovertemplate=None)
    figure.update_layout(yaxis_title=y_title, xaxis_title=None)
    return _apply_light_theme(figure)


def cluster_bar(data_frame: pd.DataFrame, title: str) -> go.Figure:
    """Build a stable cluster-size bar chart."""
    data_frame = data_frame.copy()
    data_frame["cluster"] = data_frame["cluster"].astype(str)
    figure = px.bar(
        data_frame,
        x="cluster",
        y="cluster_size",
        color="cluster",
        color_discrete_map={str(key): value for key, value in CLUSTER_COLORS.items()},
        title=title,
        text_auto=True,
    )
    figure.update_layout(showlegend=False, xaxis_title="Cluster")
    return _apply_light_theme(figure)


def heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    """Build a correlation heatmap."""
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        )
    )
    figure.update_layout(title=title, height=620)
    return _apply_light_theme(figure)


def scatter_chart(
    data_frame: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
) -> go.Figure:
    """Build a cluster scatter chart."""
    data_frame = data_frame.copy()
    data_frame[color] = data_frame[color].astype(str)
    figure = px.scatter(
        data_frame,
        x=x,
        y=y,
        color=color,
        color_discrete_map={str(key): value for key, value in CLUSTER_COLORS.items()},
        hover_data=[column for column in ["Customer_ID", "cluster"] if column in data_frame],
        title=title,
        opacity=0.65,
    )
    figure.update_layout(legend_title_text="Cluster")
    return _apply_light_theme(figure)
