import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import rasterio
except ImportError:
    rasterio = None

st.set_page_config(
    page_title="FloodLens | Pakistan Earth-System Trends",
    page_icon="🌧️",
    layout="wide",
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"

MONTHLY_FILE = DATA / "Pakistan_GPM_Rainfall_2001_2024.csv"
SPATIAL_FILE = DATA / "FloodLens_Pakistan_Spatial_Rainfall_2001_2024.csv"
TRENDS_FILE = DATA / "FloodLens_Pakistan_Pixel_Trends_2001_2024.csv"
BOUNDARY_FILE = DATA / "pakistan_boundary.geojson"
TERRAIN_FILE = DATA / "FloodLens_Pakistan_Terrain_1km.tif"


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data
def read_boundary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_boundary_lines(fig, geojson, color="#222222", width=1.4):
    """Add Pakistan's GeoJSON outline to a Plotly longitude/latitude figure."""
    if geojson.get("type") == "FeatureCollection":
        geometries = [
            feature.get("geometry")
            for feature in geojson.get("features", [])
            if feature.get("geometry")
        ]
    elif geojson.get("type") == "Feature":
        geometries = [geojson.get("geometry")]
    else:
        geometries = [geojson]

    def add_ring(ring):
        if not ring:
            return
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip", showlegend=False
            )
        )

    for geometry in geometries:
        if not geometry:
            continue
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates", [])
        if kind == "Polygon":
            for ring in coordinates:
                add_ring(ring)
        elif kind == "MultiPolygon":
            for polygon in coordinates:
                for ring in polygon:
                    add_ring(ring)
        elif kind == "GeometryCollection":
            for subgeometry in geometry.get("geometries", []):
                subkind = subgeometry.get("type")
                subcoords = subgeometry.get("coordinates", [])
                if subkind == "Polygon":
                    for ring in subcoords:
                        add_ring(ring)
                elif subkind == "MultiPolygon":
                    for polygon in subcoords:
                        for ring in polygon:
                            add_ring(ring)


def load_monthly():
    if not MONTHLY_FILE.exists():
        return None
    df = read_csv(MONTHLY_FILE)
    if "date" not in df or "monthly_rainfall_mm" not in df:
        st.error(
            "Monthly CSV needs columns named 'date' and "
            "'monthly_rainfall_mm'."
        )
        return None
    df = df[["date", "monthly_rainfall_mm"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["monthly_rainfall_mm"] = pd.to_numeric(
        df["monthly_rainfall_mm"], errors="coerce"
    )
    return df.dropna().sort_values("date")


def load_trends():
    if not TRENDS_FILE.exists():
        return None
    df = read_csv(TRENDS_FILE)
    required = {"latitude", "longitude", "sen_slope_mm_year"}
    if not required.issubset(df.columns):
        st.error(
            "Trend CSV must include latitude, longitude and "
            "sen_slope_mm_year columns."
        )
        return None
    for col in ["latitude", "longitude", "sen_slope_mm_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    p_col = "adjusted_p"
    if p_col not in df.columns and "adjusted_p_value" in df.columns:
        p_col = "adjusted_p_value"
    if p_col in df.columns:
        df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
        df["significant_fdr"] = df[p_col] < 0.05
    elif "significant" in df.columns:
        df["significant_fdr"] = df["significant"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
    else:
        df["significant_fdr"] = False
    return df.dropna(subset=["latitude", "longitude", "sen_slope_mm_year"])


def make_trend_map(trends, boundary, mode):
    plot_df = trends.copy()
    if mode == "Statistical significance":
        plot_df["status"] = np.where(
            plot_df["significant_fdr"], "Significant after FDR",
            "Not significant after FDR"
        )
        fig = px.scatter(
            plot_df, x="longitude", y="latitude",
            color="status",
            color_discrete_map={
                "Significant after FDR": "#147D42",
                "Not significant after FDR": "#E58B83",
            },
            hover_data={
                "sen_slope_mm_year": ":.2f",
                "latitude": ":.2f",
                "longitude": ":.2f",
            },
            labels={
                "longitude": "Longitude",
                "latitude": "Latitude",
                "status": "Statistical result",
                "sen_slope_mm_year": "Sen's slope (mm/year)",
            },
        )
    else:
        fig = px.scatter(
            plot_df, x="longitude", y="latitude",
            color="sen_slope_mm_year",
            color_continuous_scale="RdYlBu_r",
            range_color=(-8, 16),
            hover_data={
                "sen_slope_mm_year": ":.2f",
                "latitude": ":.2f",
                "longitude": ":.2f",
                "significant_fdr": True,
            },
            labels={
                "longitude": "Longitude",
                "latitude": "Latitude",
                "sen_slope_mm_year": "Sen's slope (mm/year)",
                "significant_fdr": "Significant after FDR",
            },
        )
    if boundary is not None:
        add_boundary_lines(fig, boundary)
    fig.update_traces(marker=dict(size=5, opacity=0.85))
    fig.update_layout(
        height=720,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(range=[60, 78], title="Longitude", scaleanchor="y"),
        yaxis=dict(range=[22, 38], title="Latitude"),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def show_missing_files():
    missing = []
    for path in [MONTHLY_FILE, TRENDS_FILE, BOUNDARY_FILE]:
        if not path.exists():
            missing.append(path.name)
    if missing:
        st.warning(
            "Some data files are not in the data/ folder yet: "
            + ", ".join(missing)
            + ". See README.md for the Colab export steps."
        )


st.title("🌧️ FloodLens")
st.caption(
    "Pakistan Earth-system trend explorer | GPM IMERG monthly rainfall "
    "and SRTM elevation"
)

with st.sidebar:
    st.header("Explore FloodLens")
    page = st.radio(
        "Dashboard section",
        [
            "National overview",
            "Rainfall explorer",
            "Spatial trend explorer",
            "Terrain explorer",
            "About & data",
        ],
    )
    st.divider()
    st.caption("Study period: 2001–2024")
    st.caption("Study area: Pakistan")

show_missing_files()
monthly = load_monthly()
trends = load_trends()
boundary = read_boundary(BOUNDARY_FILE) if BOUNDARY_FILE.exists() else None

if page == "National overview":
    st.subheader("National rainfall overview")
    if monthly is None:
        st.info("Add the monthly rainfall CSV to data/ to populate this page.")
    else:
        monthly = monthly.copy()
        monthly["year"] = monthly["date"].dt.year
        annual = (
            monthly.groupby("year", as_index=False)["monthly_rainfall_mm"]
            .sum()
            .rename(columns={"monthly_rainfall_mm": "annual_rainfall_mm"})
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Monthly observations", f"{len(monthly):,}")
        c2.metric("Years covered", f"{monthly['date'].dt.year.min()}–{monthly['date'].dt.year.max()}")
        c3.metric("Mean annual rainfall", f"{annual['annual_rainfall_mm'].mean():,.1f} mm")
        fig = px.line(
            annual, x="year", y="annual_rainfall_mm",
            markers=True, title="Pakistan annual rainfall total"
        )
        fig.update_layout(xaxis_title="Year", yaxis_title="Annual rainfall (mm)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### National trend result")
        a, b, c = st.columns(3)
        a.metric("Trend direction", "Increasing")
        b.metric("Hamed–Rao p-value", "0.035")
        c.metric("Sen's slope", "+5.231 mm/year")
        st.caption(
            "Preliminary national result from the current analysis. "
            "It does not by itself establish the cause of the trend or "
            "prove that every region is becoming wetter."
        )

elif page == "Rainfall explorer":
    st.subheader("Rainfall explorer")
    if monthly is None:
        st.info("Add the monthly rainfall CSV to data/ to populate this page.")
    else:
        monthly = monthly.copy()
        monthly["year"] = monthly["date"].dt.year
        monthly["month"] = monthly["date"].dt.month
        min_year = int(monthly["year"].min())
        max_year = int(monthly["year"].max())
        selected_years = st.slider(
            "Select year range", min_year, max_year, (min_year, max_year)
        )
        subset = monthly[
            monthly["year"].between(selected_years[0], selected_years[1])
        ]
        fig = px.line(
            subset, x="date", y="monthly_rainfall_mm",
            title="Monthly rainfall through time"
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Rainfall (mm)")
        st.plotly_chart(fig, use_container_width=True)

        heat = subset.pivot_table(
            index="year", columns="month",
            values="monthly_rainfall_mm", aggfunc="mean"
        )
        fig_heat = px.imshow(
            heat, aspect="auto", origin="lower",
            color_continuous_scale="Blues",
            labels=dict(x="Month", y="Year", color="Rainfall (mm)"),
            title="Monthly rainfall heatmap"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        baseline = monthly[monthly["year"].between(2001, 2020)].copy()
        baseline["month"] = baseline["date"].dt.month
        monthly["calendar_month"] = monthly["date"].dt.month
        climatology = baseline.groupby("month")["monthly_rainfall_mm"].mean()
        monthly["anomaly_mm"] = (
            monthly["monthly_rainfall_mm"]
            - monthly["calendar_month"].map(climatology)
        )
        anomaly_subset = monthly[
            monthly["year"].between(selected_years[0], selected_years[1])
        ]
        fig_anom = px.bar(
            anomaly_subset, x="date", y="anomaly_mm",
            title="Monthly rainfall anomalies (relative to 2001–2020 calendar-month mean)"
        )
        fig_anom.update_layout(xaxis_title="Month", yaxis_title="Anomaly (mm)")
        st.plotly_chart(fig_anom, use_container_width=True)

elif page == "Spatial trend explorer":
    st.subheader("Spatial rainfall trend explorer")
    if trends is None:
        st.info(
            "Add FloodLens_Pakistan_Pixel_Trends_2001_2024.csv to data/. "
            "This file contains each grid cell's Sen's slope and adjusted p-value."
        )
    else:
        mode = st.radio(
            "Map layer",
            ["Sen's slope (mm/year)", "Statistical significance"],
            horizontal=True,
        )
        st.plotly_chart(
            make_trend_map(trends, boundary, mode),
            use_container_width=True,
        )
        st.write(f"Grid cells shown: **{len(trends):,}**")
        if "adjusted_p" in trends.columns:
            st.write(
                "Significant after FDR (adjusted p < 0.05): "
                f"**{int(trends['significant_fdr'].sum()):,}**"
            )
        st.caption(
            "Sen's slope describes the estimated direction and rate of change. "
            "Statistical significance is shown separately; a coloured slope "
            "does not mean the trend is statistically significant."
        )
        with st.expander("Inspect a grid cell"):
            lat = st.number_input(
                "Latitude", min_value=24.0, max_value=37.0, value=30.0, step=0.1
            )
            lon = st.number_input(
                "Longitude", min_value=61.0, max_value=76.0, value=70.0, step=0.1
            )
            nearest_index = (
                (trends["latitude"] - lat) ** 2
                + (trends["longitude"] - lon) ** 2
            ).idxmin()
            point = trends.loc[nearest_index]
            st.write(f"Nearest grid cell: {point['latitude']:.2f}°N, {point['longitude']:.2f}°E")
            st.metric("Sen's slope", f"{point['sen_slope_mm_year']:.2f} mm/year")
            if "adjusted_p" in point.index:
                st.metric("FDR-adjusted p-value", f"{point['adjusted_p']:.4f}")
            st.write(
                "Significant after FDR: "
                + ("Yes" if bool(point["significant_fdr"]) else "No")
            )

elif page == "Terrain explorer":
    st.subheader("Pakistan terrain explorer")
    if not TERRAIN_FILE.exists():
        st.info(
            "Optional terrain file not found. Add "
            "FloodLens_Pakistan_Terrain_1km.tif to data/ to enable this page."
        )
    elif rasterio is None:
        st.error("Install rasterio with: pip install rasterio")
    else:
        with rasterio.open(TERRAIN_FILE) as src:
            elevation = src.read(1).astype("float32")
            nodata = src.nodata
            bounds = src.bounds
        if nodata is not None:
            elevation[elevation == nodata] = np.nan
        max_side = 140
        step = max(1, int(max(elevation.shape) / max_side))
        small = elevation[::step, ::step]
        st.metric("Raster elevation range", f"{np.nanmin(elevation):.0f} to {np.nanmax(elevation):.0f} m")
        fig = px.imshow(
            small, origin="upper", aspect="auto",
            color_continuous_scale="Earth",
            title="Pakistan digital elevation model (SRTM)"
        )
        fig.update_layout(coloraxis_colorbar_title="Elevation (m)")
        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show interactive 3D terrain surface"):
            rows, cols = small.shape
            x = np.linspace(bounds.left, bounds.right, cols)[::1]
            y = np.linspace(bounds.top, bounds.bottom, rows)[::1]
            X, Y = np.meshgrid(x, y)
            fig3d = go.Figure(
                data=[go.Surface(x=X, y=Y, z=small, colorscale="Earth")]
            )
            fig3d.update_layout(
                title="3D terrain surface",
                scene=dict(
                    xaxis_title="Easting (m)",
                    yaxis_title="Northing (m)",
                    zaxis_title="Elevation (m)",
                    aspectratio=dict(x=1, y=1, z=0.35),
                ),
                height=700,
            )
            st.plotly_chart(fig3d, use_container_width=True)

elif page == "About & data":
    st.subheader("About FloodLens")
    st.write(
        "FloodLens is an exploratory Earth-system trend detection platform. "
        "Its initial Pakistan case study uses NASA GPM IMERG monthly "
        "precipitation data (2001–2024), with SRTM elevation as static "
        "geographical context."
    )
    st.markdown(
        """
        **Scientific interpretation**
        - Rainfall trends are estimates, not flood forecasts.
        - The national aggregate trend and local pixel trends are different analyses.
        - The original Mann–Kendall pixel tests found no grid cells significant
          after Benjamini–Hochberg false-discovery-rate correction.
        - Elevation is static context, not a time-varying trend variable.
        - Satellite/model rainfall estimates have uncertainty and spatial resolution limits.
        """
    )
    st.markdown("**Data files expected in `data/`**")
    st.code(
        "Pakistan_GPM_Rainfall_2001_2024.csv\n"
        "FloodLens_Pakistan_Pixel_Trends_2001_2024.csv\n"
        "pakistan_boundary.geojson\n"
        "FloodLens_Pakistan_Terrain_1km.tif  # optional"
    )
    st.markdown(
        "Primary rainfall source: NASA GPM IMERG Monthly V07. "
        "Terrain source: USGS SRTMGL1."
    )
