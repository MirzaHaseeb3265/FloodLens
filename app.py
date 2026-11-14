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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodLens | Pakistan Earth-System Trends",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------
ACCENT_CYAN = "#00E5FF"
ACCENT_ORANGE = "#FC3D21"
ACCENT_NASA_BLUE = "#0B3D91"
TEXT_LIGHT = "#E6F1FF"
TEXT_MUTED = "#8CA3C4"
GRID_FAINT = "rgba(255,255,255,0.08)"


def inject_css():
    """Inject the whole mission-control theme in one place."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        :root {
            --bg-0: #050A18;
            --bg-1: #0B1226;
            --panel: rgba(10, 22, 48, 0.55);
            --panel-border: rgba(0, 229, 255, 0.18);
            --cyan: #00E5FF;
            --orange: #FC3D21;
            --nasa-blue: #0B3D91;
            --text: #E6F1FF;
            --muted: #8CA3C4;
        }

        /* -------- App background: deep-space gradient + faint star grid -------- */
        .stApp {
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(11,61,145,0.35), transparent 60%),
                radial-gradient(900px 500px at 110% 20%, rgba(0,229,255,0.10), transparent 60%),
                linear-gradient(160deg, #050A18 0%, #0B1226 100%);
            background-attachment: fixed;
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.55), transparent 60%),
                radial-gradient(1px 1px at 70% 60%, rgba(255,255,255,0.45), transparent 60%),
                radial-gradient(1px 1px at 40% 80%, rgba(0,229,255,0.55), transparent 60%),
                radial-gradient(1px 1px at 85% 15%, rgba(255,255,255,0.40), transparent 60%),
                radial-gradient(1px 1px at 10% 75%, rgba(255,255,255,0.40), transparent 60%);
            background-size: 400px 400px;
            opacity: 0.5;
            pointer-events: none;
            z-index: 0;
        }
        .stApp::after {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(0,229,255,0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,229,255,0.035) 1px, transparent 1px);
            background-size: 64px 64px;
            pointer-events: none;
            z-index: 0;
        }

        /* -------- Hide default Streamlit chrome -------- */
        #MainMenu, header, footer { visibility: hidden; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1350px; }

        /* -------- Typography -------- */
        h1, h2, h3, h4, h5, h6, .stSubheader {
            font-family: 'Orbitron', 'Rajdhani', sans-serif !important;
            letter-spacing: 0.06em;
            color: var(--text) !important;
        }
        p, span, label, div, .stMarkdown { color: var(--text); }

        /* -------- Hero header -------- */
        .fl-hero {
            position: relative;
            padding: 26px 30px 22px 30px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(11,61,145,0.35), rgba(0,229,255,0.06));
            border: 1px solid var(--panel-border);
            box-shadow: 0 0 30px rgba(0,229,255,0.10), inset 0 0 24px rgba(0,229,255,0.04);
            overflow: hidden;
            animation: fadeIn 0.8s ease-out;
        }
        .fl-hero::before {
            content: "";
            position: absolute;
            inset: -50% -20% auto auto;
            width: 500px; height: 500px;
            background: radial-gradient(circle, rgba(0,229,255,0.18), transparent 60%);
            pointer-events: none;
        }
        .fl-title {
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            font-size: 44px;
            letter-spacing: 0.32em;
            color: var(--text);
            text-shadow: 0 0 18px rgba(0,229,255,0.55), 0 0 4px rgba(0,229,255,0.9);
            margin: 0;
            line-height: 1.05;
        }
        .fl-sub {
            font-family: 'Rajdhani', sans-serif;
            font-weight: 500;
            font-size: 15px;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin-top: 8px;
            text-transform: uppercase;
        }
        .fl-divider {
            margin: 16px 0 12px 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--cyan), transparent);
            box-shadow: 0 0 12px rgba(0,229,255,0.7);
            border: none;
        }
        .fl-chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
        .fl-chip {
            font-family: 'Rajdhani', sans-serif;
            font-size: 12px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(0,229,255,0.35);
            background: rgba(0,229,255,0.06);
            color: var(--cyan);
            box-shadow: 0 0 12px rgba(0,229,255,0.15);
        }
        .fl-chip.orange {
            border-color: rgba(252,61,33,0.35);
            background: rgba(252,61,33,0.06);
            color: #FF8A73;
        }

        /* -------- Glass cards -------- */
        .fl-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 20px 22px;
            box-shadow: 0 0 24px rgba(0,229,255,0.06), inset 0 0 18px rgba(0,229,255,0.03);
            backdrop-filter: blur(8px);
            margin-bottom: 18px;
            animation: fadeIn 0.7s ease-out;
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .fl-card:hover {
            border-color: rgba(0,229,255,0.4);
            box-shadow: 0 0 32px rgba(0,229,255,0.15), inset 0 0 22px rgba(0,229,255,0.05);
        }
        .fl-card-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 13px;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--cyan);
            margin-bottom: 14px;
        }

        /* -------- HUD metric cards (override st.metric) -------- */
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-left: 3px solid var(--cyan);
            border-radius: 12px;
            padding: 14px 16px 12px 16px;
            box-shadow: 0 0 18px rgba(0,229,255,0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            animation: fadeIn 0.6s ease-out;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 28px rgba(0,229,255,0.22);
            border-left-color: var(--orange);
        }
        div[data-testid="stMetricLabel"] > div {
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 11px !important;
            letter-spacing: 0.18em !important;
            text-transform: uppercase !important;
            color: var(--muted) !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 26px !important;
            font-weight: 700 !important;
            color: var(--cyan) !important;
            text-shadow: 0 0 12px rgba(0,229,255,0.45);
        }

        /* -------- Sidebar: mission control menu -------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #070E1F 0%, #050A18 100%);
            border-right: 1px solid rgba(0,229,255,0.15);
            box-shadow: inset -1px 0 12px rgba(0,229,255,0.05);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: 'Orbitron', sans-serif !important;
            color: var(--text) !important;
            letter-spacing: 0.14em;
        }
        .fl-sidebar-logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 22px;
            font-weight: 900;
            letter-spacing: 0.28em;
            color: var(--text);
            text-shadow: 0 0 14px rgba(0,229,255,0.6);
            margin-bottom: 2px;
        }
        .fl-sidebar-tag {
            font-family: 'Rajdhani', sans-serif;
            font-size: 11px;
            letter-spacing: 0.22em;
            color: var(--cyan);
            text-transform: uppercase;
            margin-bottom: 18px;
        }
        /* Radio as menu */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            display: block;
            padding: 10px 14px;
            margin: 4px 0;
            border-radius: 10px;
            border: 1px solid transparent;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.06em;
            color: var(--text);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: rgba(0,229,255,0.06);
            border-color: rgba(0,229,255,0.35);
            box-shadow: 0 0 16px rgba(0,229,255,0.15);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
            background: linear-gradient(90deg, rgba(0,229,255,0.16), rgba(0,229,255,0.02));
            border-color: var(--cyan);
            box-shadow: 0 0 20px rgba(0,229,255,0.28), inset 0 0 14px rgba(0,229,255,0.08);
            color: var(--cyan) !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] * {
            color: var(--cyan) !important;
        }
        /* Hide default radio dot in sidebar */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
            display: none;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] small {
            color: var(--muted) !important;
            letter-spacing: 0.08em;
        }

        /* -------- Inputs & controls -------- */
        .stSlider label, .stSelectbox label, .stNumberInput label,
        .stRadio > label, .stCheckbox label, .stTextInput label {
            font-family: 'Rajdhani', sans-serif !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
            font-size: 12px !important;
            color: var(--muted) !important;
        }
        div[data-baseweb="slider"] > div > div > div > div {
            background: linear-gradient(90deg, var(--cyan), var(--nasa-blue)) !important;
        }
        .stButton>button, .stDownloadButton>button {
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.1em;
            background: rgba(0,229,255,0.08);
            border: 1px solid rgba(0,229,255,0.4);
            color: var(--cyan);
            border-radius: 10px;
        }
        .stButton>button:hover {
            background: rgba(0,229,255,0.18);
            box-shadow: 0 0 18px rgba(0,229,255,0.3);
            color: #fff;
        }
        .stExpander {
            border: 1px solid var(--panel-border) !important;
            border-radius: 12px !important;
            background: rgba(10,22,48,0.4) !important;
        }
        .stExpander summary {
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.1em;
            color: var(--cyan) !important;
        }
        code, pre {
            background: rgba(0,0,0,0.35) !important;
            color: #8CF8FF !important;
            border-radius: 8px;
        }
        hr { border-color: rgba(0,229,255,0.15); }

        /* -------- HUD caption under charts -------- */
        .fl-hud-caption {
            font-family: 'Rajdhani', sans-serif;
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted);
            margin: -8px 0 18px 2px;
            padding-left: 10px;
            border-left: 2px solid var(--cyan);
        }

        /* -------- Data source badges -------- */
        .fl-badge {
            display: inline-block;
            padding: 8px 14px;
            margin: 6px 8px 0 0;
            border-radius: 10px;
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.1em;
            font-size: 13px;
            border: 1px solid rgba(0,229,255,0.35);
            background: rgba(0,229,255,0.06);
            color: var(--cyan);
            box-shadow: 0 0 12px rgba(0,229,255,0.15);
        }
        .fl-badge.orange {
            border-color: rgba(252,61,33,0.4);
            background: rgba(252,61,33,0.08);
            color: #FF8A73;
            box-shadow: 0 0 12px rgba(252,61,33,0.15);
        }

        /* -------- Section heading -------- */
        .fl-section {
            font-family: 'Orbitron', sans-serif;
            font-size: 14px;
            letter-spacing: 0.24em;
            text-transform: uppercase;
            color: var(--cyan);
            margin: 6px 0 14px 2px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(0,229,255,0.2);
        }

        /* -------- Animations -------- */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* -------- Responsive -------- */
        @media (max-width: 900px) {
            .fl-title { font-size: 30px; letter-spacing: 0.2em; }
            .fl-sub { font-size: 12px; }
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Plotly theming helper
# ---------------------------------------------------------------------------
def style_fig(fig, height=None, title=None):
    """Apply the mission-control look to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_LIGHT, size=12),
        title=dict(
            text=title if title is not None else (fig.layout.title.text or ""),
            font=dict(family="Orbitron, sans-serif", size=15, color=TEXT_LIGHT),
            x=0.01, xanchor="left",
        ),
        margin=dict(l=20, r=20, t=55, b=30),
        hoverlabel=dict(
            bgcolor="#081A2E",
            bordercolor=ACCENT_CYAN,
            font=dict(family="Inter, sans-serif", color=TEXT_LIGHT, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,229,255,0.2)",
            font=dict(color=TEXT_LIGHT),
        ),
    )
    # Grid styling for Cartesian axes
    axis_style = dict(
        showgrid=True,
        gridcolor=GRID_FAINT,
        zeroline=False,
        linecolor="rgba(255,255,255,0.15)",
        tickfont=dict(color=TEXT_LIGHT),
        title_font=dict(color=TEXT_LIGHT, family="Rajdhani, sans-serif", size=12),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    # Colorbars
    for trace in fig.data:
        if getattr(trace, "colorbar", None) is not None:
            trace.colorbar.tickfont = dict(color=TEXT_LIGHT)
            trace.colorbar.title.font = dict(color=TEXT_LIGHT, size=12)
            trace.colorbar.outlinecolor = "rgba(255,255,255,0.15)"
            trace.colorbar.bgcolor = "rgba(0,0,0,0)"
    if height is not None:
        fig.update_layout(height=height)
    return fig


def hud_caption(text):
    """Small HUD-style caption under a chart."""
    st.markdown(f'<div class="fl-hud-caption">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data paths & loaders (unchanged logic)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA = ROOT / "data"

MONTHLY_FILE = DATA / "Pakistan_GPM_Rainfall_2001_2024.csv"
SPATIAL_FILE = DATA / "FloodLens_Pakistan_Spatial_Rainfall_2001_2024.csv"
TRENDS_FILE = DATA / "FloodLens_Pakistan_Pixel_Trends_2001_2024.csv"
BOUNDARY_FILE = DATA / "pakistan_boundary.geojson"
TERRAIN_FILE = DATA / "FloodLens_Pakistan_Terrain_1km.tif"
FORECAST_DATA_FILE = DATA / "FloodLens_Regional_Forecast_Dataset.csv"
FORECAST_VALIDATION_FILE = DATA / "FloodLens_Forecast_Validation.csv"
FORECAST_PERFORMANCE_FILE = DATA / "FloodLens_Final_Regional_Performance.csv"


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_forecast_files():
    """Load historical regional forecast experiment and validation outputs."""
    required_files = [
        FORECAST_DATA_FILE,
        FORECAST_VALIDATION_FILE,
        FORECAST_PERFORMANCE_FILE,
    ]
    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        return None, None, None, missing

    dataset = pd.read_csv(FORECAST_DATA_FILE)
    validation = pd.read_csv(FORECAST_VALIDATION_FILE)
    performance = pd.read_csv(FORECAST_PERFORMANCE_FILE)

    return dataset, validation, performance, []


def make_forecast_template(regions, forecast_year):
    """Create a blank upload template for one future monsoon season."""
    return pd.DataFrame({
        "year": [int(forecast_year)] * len(regions),
        "region": list(regions),
        "pre_rainfall_mm": [np.nan] * len(regions),
        "pre_soil_moisture_mm": [np.nan] * len(regions),
    })


def run_regional_forecast(history, performance, future_inputs, alpha=10.0):
    """Fit the historical Ridge experiment and produce regional estimates.

    Numeric predictors are standardized; region is one-hot encoded. The
    intercept is unpenalized and Ridge coefficients use alpha=10, matching
    the original experiment's preprocessing/model setup.
    """
    feature_cols = ["pre_rainfall_mm", "pre_soil_moisture_mm"]
    target_col = "monsoon_rainfall_mm"
    regions = sorted(history["region"].dropna().unique().tolist())

    train = history.dropna(subset=feature_cols + [target_col]).copy()
    if train.empty:
        raise ValueError("The historical training dataset has no complete rows.")

    x_num = train[feature_cols].astype(float).to_numpy()
    x_mean = x_num.mean(axis=0)
    x_std = x_num.std(axis=0, ddof=0)
    x_std[x_std == 0] = 1.0
    x_num_scaled = (x_num - x_mean) / x_std

    train_regions = train["region"].astype(str).tolist()
    onehot = np.array([[1.0 if r == category else 0.0 for category in regions]
                       for r in train_regions], dtype=float)
    x_train = np.column_stack([x_num_scaled, onehot])
    y_train = train[target_col].astype(float).to_numpy()

    x_bar = x_train.mean(axis=0)
    y_bar = y_train.mean()
    x_centered = x_train - x_bar
    y_centered = y_train - y_bar
    penalty = np.eye(x_train.shape[1], dtype=float) * float(alpha)
    coefficients = np.linalg.solve(x_centered.T @ x_centered + penalty,
                                   x_centered.T @ y_centered)
    intercept = y_bar - x_bar @ coefficients

    future = future_inputs.copy()
    future["region"] = future["region"].astype(str)
    x_future_num = future[feature_cols].astype(float).to_numpy()
    x_future_scaled = (x_future_num - x_mean) / x_std
    x_future_onehot = np.array([
        [1.0 if r == category else 0.0 for category in regions]
        for r in future["region"].tolist()
    ], dtype=float)
    x_future = np.column_stack([x_future_scaled, x_future_onehot])
    future["ridge_prediction_mm"] = intercept + x_future @ coefficients

    regional_baseline = train.groupby("region")[target_col].mean().to_dict()
    future["baseline_prediction_mm"] = future["region"].map(regional_baseline)

    method_map = performance.set_index("region")["Recommended_method"].astype(str).to_dict()
    future["recommended_method"] = future["region"].map(method_map).fillna("Experimental Ridge")
    future["selected_prediction_mm"] = np.where(
        future["recommended_method"].str.contains("Ridge", case=False, na=False),
        future["ridge_prediction_mm"],
        future["baseline_prediction_mm"],
    )
    future["selected_prediction_mm"] = future["selected_prediction_mm"].clip(lower=0)
    future["ridge_prediction_mm"] = future["ridge_prediction_mm"].clip(lower=0)
    future["baseline_prediction_mm"] = future["baseline_prediction_mm"].clip(lower=0)
    return future


@st.cache_data
def read_boundary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_boundary_lines(fig, geojson, color="#00E5FF", width=1.4):
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
                hoverinfo="skip", showlegend=False,
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
                "Significant after FDR": "#00FF9C",
                "Not significant after FDR": "#FF6B7A",
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
            color_continuous_scale="Turbo",
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
    fig.update_traces(marker=dict(size=7, opacity=0.9, line=dict(width=0.5, color="rgba(255,255,255,0.35)")))
    fig.update_layout(
        height=720,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(range=[60, 78], title="Longitude", scaleanchor="y"),
        yaxis=dict(range=[22, 38], title="Latitude"),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return style_fig(fig, height=720)


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


# ---------------------------------------------------------------------------
# Boot the theme
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="fl-hero">
        <div class="fl-title">🛰️ FLOODLENS</div>
        <div class="fl-sub">Pakistan Earth-System Trend Explorer | NASA GPM IMERG &amp; SRTM</div>
        <hr class="fl-divider" />
        <div class="fl-chips">
            <span class="fl-chip">STUDY PERIOD 2001–2024</span>
            <span class="fl-chip">REGION: PAKISTAN</span>
            <span class="fl-chip orange">MISSION: EARTH OBSERVATION</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: mission-control menu
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="fl-sidebar-logo">FLOODLENS</div>
        <div class="fl-sidebar-tag">Mission Control</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Dashboard section",
        [
            "National overview",
            "Rainfall explorer",
            "Spatial trend explorer",
            "Terrain explorer",
            "Regional monsoon outlook",
            "About & data",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Study period: 2001–2024")
    st.caption("Study area: Pakistan")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
show_missing_files()
monthly = load_monthly()
trends = load_trends()
boundary = read_boundary(BOUNDARY_FILE) if BOUNDARY_FILE.exists() else None


# ---------------------------------------------------------------------------
# PAGE: National overview
# ---------------------------------------------------------------------------
if page == "National overview":
    st.markdown('<div class="fl-section">National Overview</div>', unsafe_allow_html=True)
    st.markdown("### 🌍 National rainfall overview")

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

        # ---- Telemetry cards row 1 ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Telemetry · Dataset</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Monthly observations", f"{len(monthly):,}")
        c2.metric(
            "Years covered",
            f"{monthly['date'].dt.year.min()}–{monthly['date'].dt.year.max()}",
        )
        c3.metric(
            "Mean annual rainfall",
            f"{annual['annual_rainfall_mm'].mean():,.1f} mm",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Annual rainfall line chart ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Annual Signal</div>', unsafe_allow_html=True)
        fig = px.line(
            annual, x="year", y="annual_rainfall_mm",
            markers=True, title="Pakistan annual rainfall total",
        )
        fig.update_traces(
            line=dict(color=ACCENT_CYAN, width=2.4, shape="spline"),
            marker=dict(size=8, color=ACCENT_CYAN, line=dict(width=1.5, color="white")),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.12)",
        )
        fig.update_layout(xaxis_title="Year", yaxis_title="Annual rainfall (mm)")
        fig = style_fig(fig, height=460)
        st.plotly_chart(fig, use_container_width=True)
        hud_caption("ANNUAL NATIONAL RAINFALL TOTALS · GPM IMERG MONTHLY V07")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Telemetry cards row 2: national trend ----
        st.markdown("#### National trend result")
        st.markdown('<div class="fl-card"><div class="fl-card-title">Telemetry · Trend Diagnostics</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Trend direction", "Increasing")
        b.metric("Hamed–Rao p-value", "0.035")
        c.metric("Sen's slope", "+5.231 mm/year")
        st.markdown("</div>", unsafe_allow_html=True)

        st.caption(
            "Preliminary national result from the current analysis. "
            "It does not by itself establish the cause of the trend or "
            "prove that every region is becoming wetter."
        )


# ---------------------------------------------------------------------------
# PAGE: Rainfall explorer
# ---------------------------------------------------------------------------
elif page == "Rainfall explorer":
    st.markdown('<div class="fl-section">Rainfall Explorer</div>', unsafe_allow_html=True)
    st.markdown("### 📡 Rainfall explorer")

    if monthly is None:
        st.info("Add the monthly rainfall CSV to data/ to populate this page.")
    else:
        monthly = monthly.copy()
        monthly["year"] = monthly["date"].dt.year
        monthly["month"] = monthly["date"].dt.month
        min_year = int(monthly["year"].min())
        max_year = int(monthly["year"].max())

        # ---- Control panel card ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Control Panel</div>', unsafe_allow_html=True)
        selected_years = st.slider(
            "Select year range", min_year, max_year, (min_year, max_year)
        )
        st.markdown("</div>", unsafe_allow_html=True)

        subset = monthly[
            monthly["year"].between(selected_years[0], selected_years[1])
        ]

        # ---- Monthly time series ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Monthly Signal</div>', unsafe_allow_html=True)
        fig = px.line(
            subset, x="date", y="monthly_rainfall_mm",
            title="Monthly rainfall through time",
        )
        fig.update_traces(
            line=dict(color=ACCENT_CYAN, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(0,229,255,0.10)",
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Rainfall (mm)")
        fig = style_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
        hud_caption("MONTHLY RAINFALL THROUGH TIME · GPM IMERG")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Heatmap ----
        heat = subset.pivot_table(
            index="year", columns="month",
            values="monthly_rainfall_mm", aggfunc="mean",
        )
        st.markdown('<div class="fl-card"><div class="fl-card-title">Seasonal Matrix</div>', unsafe_allow_html=True)
        fig_heat = px.imshow(
            heat, aspect="auto", origin="lower",
            color_continuous_scale=[
                [0.0, "#050A18"],
                [0.25, "#0B3D91"],
                [0.55, "#00A6C0"],
                [0.8, "#00E5FF"],
                [1.0, "#FFFFFF"],
            ],
            labels=dict(x="Month", y="Year", color="Rainfall (mm)"),
            title="Monthly rainfall heatmap",
        )
        fig_heat = style_fig(fig_heat, height=480)
        st.plotly_chart(fig_heat, use_container_width=True)
        hud_caption("CALENDAR-MONTH MEAN RAINFALL · DEEP-NAVY → CYAN → WHITE SCALE")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Anomalies ----
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

        st.markdown('<div class="fl-card"><div class="fl-card-title">Anomaly Pulse</div>', unsafe_allow_html=True)
        fig_anom = px.bar(
            anomaly_subset, x="date", y="anomaly_mm",
            title="Monthly rainfall anomalies (relative to 2001–2020 calendar-month mean)",
        )
        # Positive cyan, negative orange/red
        fig_anom.update_traces(
            marker=dict(
                color=np.where(
                    anomaly_subset["anomaly_mm"] >= 0,
                    ACCENT_CYAN, ACCENT_ORANGE,
                ),
                line=dict(width=0),
            ),
        )
        fig_anom.update_layout(xaxis_title="Month", yaxis_title="Anomaly (mm)")
        fig_anom = style_fig(fig_anom, height=430)
        st.plotly_chart(fig_anom, use_container_width=True)
        hud_caption("POSITIVE ANOMALIES = CYAN · NEGATIVE ANOMALIES = ORANGE")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Spatial trend explorer
# ---------------------------------------------------------------------------
elif page == "Spatial trend explorer":
    st.markdown('<div class="fl-section">Spatial Trend Explorer</div>', unsafe_allow_html=True)
    st.markdown("### 🌐 Spatial rainfall trend explorer")

    if trends is None:
        st.info(
            "Add FloodLens_Pakistan_Pixel_Trends_2001_2024.csv to data/. "
            "This file contains each grid cell's Sen's slope and adjusted p-value."
        )
    else:
        # ---- Control panel ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Control Panel</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Map layer",
            ["Sen's slope (mm/year)", "Statistical significance"],
            horizontal=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Map ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Pixel Map</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_trend_map(trends, boundary, mode),
            use_container_width=True,
        )
        hud_caption("PIXEL-LEVEL SEN'S SLOPE · PAKISTAN BOUNDARY OVERLAY")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Telemetry ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Telemetry · Pixel Counts</div>', unsafe_allow_html=True)
        st.write(f"Grid cells shown: **{len(trends):,}**")
        if "adjusted_p" in trends.columns:
            st.write(
                "Significant after FDR (adjusted p < 0.05): "
                f"**{int(trends['significant_fdr'].sum()):,}**"
            )
        st.markdown("</div>", unsafe_allow_html=True)

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
            st.write(
                f"Nearest grid cell: {point['latitude']:.2f}°N, "
                f"{point['longitude']:.2f}°E"
            )
            st.metric("Sen's slope", f"{point['sen_slope_mm_year']:.2f} mm/year")
            if "adjusted_p" in point.index:
                st.metric("FDR-adjusted p-value", f"{point['adjusted_p']:.4f}")
            st.write(
                "Significant after FDR: "
                + ("Yes" if bool(point["significant_fdr"]) else "No")
            )


# ---------------------------------------------------------------------------
# PAGE: Terrain explorer
# ---------------------------------------------------------------------------
elif page == "Terrain explorer":
    st.markdown('<div class="fl-section">Terrain Explorer</div>', unsafe_allow_html=True)
    st.markdown("### 🏔️ Pakistan terrain explorer")

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

        # ---- Telemetry ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Telemetry · Elevation</div>', unsafe_allow_html=True)
        st.metric(
            "Raster elevation range",
            f"{np.nanmin(elevation):.0f} to {np.nanmax(elevation):.0f} m",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- DEM heatmap ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">Digital Elevation Model</div>', unsafe_allow_html=True)
        fig = px.imshow(
            small, origin="upper", aspect="auto",
            color_continuous_scale=[
                [0.0, "#040814"],
                [0.15, "#0B3D91"],
                [0.4, "#00A6C0"],
                [0.7, "#9EFFE9"],
                [1.0, "#FFFFFF"],
            ],
            title="Pakistan digital elevation model (SRTM)",
        )
        fig.update_layout(coloraxis_colorbar_title="Elevation (m)")
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)
        hud_caption("SRTM 1-KM DEM · HIGH-TECH TERRAIN SCALE")
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- 3D terrain ----
        st.markdown('<div class="fl-card"><div class="fl-card-title">3D Terrain Surface</div>', unsafe_allow_html=True)
        if st.checkbox("Show interactive 3D terrain surface"):
            rows, cols = small.shape
            x = np.linspace(bounds.left, bounds.right, cols)[::1]
            y = np.linspace(bounds.top, bounds.bottom, rows)[::1]
            X, Y = np.meshgrid(x, y)
            fig3d = go.Figure(
                data=[go.Surface(
                    x=X, y=Y, z=small,
                    colorscale=[
                        [0.0, "#040814"],
                        [0.15, "#0B3D91"],
                        [0.4, "#00A6C0"],
                        [0.7, "#9EFFE9"],
                        [1.0, "#FFFFFF"],
                    ],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="Elevation (m)", font=dict(color=TEXT_LIGHT)),
                        tickfont=dict(color=TEXT_LIGHT),
                        outlinecolor="rgba(255,255,255,0.15)",
                    ),
                )]
            )
            fig3d.update_layout(
                title="3D terrain surface",
                scene=dict(
                    xaxis_title="Easting (m)",
                    yaxis_title="Northing (m)",
                    zaxis_title="Elevation (m)",
                    aspectratio=dict(x=1, y=1, z=0.35),
                    bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor=GRID_FAINT,
                        zerolinecolor=GRID_FAINT,
                        showbackground=False,
                        tickfont=dict(color=TEXT_LIGHT),
                        title_font=dict(color=TEXT_LIGHT),
                    ),
                    yaxis=dict(
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor=GRID_FAINT,
                        zerolinecolor=GRID_FAINT,
                        showbackground=False,
                        tickfont=dict(color=TEXT_LIGHT),
                        title_font=dict(color=TEXT_LIGHT),
                    ),
                    zaxis=dict(
                        backgroundcolor="rgba(0,0,0,0)",
                        gridcolor=GRID_FAINT,
                        zerolinecolor=GRID_FAINT,
                        showbackground=False,
                        tickfont=dict(color=TEXT_LIGHT),
                        title_font=dict(color=TEXT_LIGHT),
                    ),
                    camera=dict(
                        eye=dict(x=1.6, y=-1.6, z=0.9),
                    ),
                ),
                height=700,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT_LIGHT),
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(fig3d, use_container_width=True)
            hud_caption("INTERACTIVE 3D SURFACE · SRTM ELEVATION")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Regional monsoon outlook
# ---------------------------------------------------------------------------
elif page == "Regional monsoon outlook":
    st.markdown(
        '<div class="fl-section">Regional Monsoon Outlook</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 🌧️ Historical validation + future monsoon estimates")
    st.info(
        "This page presents a historical model experiment and backtest for "
        "2001–2021, plus an upload-driven tool for experimental future-year "
        "monsoon estimates. It does not automatically fetch current predictors "
        "or provide an official forecast or flood warning."
    )

    forecast_data, validation, performance, missing_forecast_files = load_forecast_files()

    if missing_forecast_files:
        st.warning(
            "Add these three files to the app's data/ folder to enable this page: "
            + ", ".join(missing_forecast_files)
        )
        st.code(
            "data/FloodLens_Regional_Forecast_Dataset.csv\\n"
            "data/FloodLens_Forecast_Validation.csv\\n"
            "data/FloodLens_Final_Regional_Performance.csv"
        )
    else:
        required_dataset_cols = {
            "year", "region", "pre_rainfall_mm",
            "pre_soil_moisture_mm", "monsoon_rainfall_mm"
        }
        required_validation_cols = {
            "year", "region", "actual",
            "ridge_prediction", "baseline_prediction"
        }
        required_performance_cols = {
            "region", "Ridge_MAE", "Baseline_MAE",
            "Improvement_mm", "Improvement_percent",
            "Recommended_method"
        }

        dataset_ok = required_dataset_cols.issubset(forecast_data.columns)
        validation_ok = required_validation_cols.issubset(validation.columns)
        performance_ok = required_performance_cols.issubset(performance.columns)

        if not (dataset_ok and validation_ok and performance_ok):
            st.error(
                "One or more forecast CSV files have unexpected column names. "
                "Please use the three files exported from the FloodLens Colab experiment."
            )
        else:
            for frame, numeric_cols in [
                (forecast_data, [
                    "year", "pre_rainfall_mm",
                    "pre_soil_moisture_mm", "monsoon_rainfall_mm"
                ]),
                (validation, [
                    "year", "actual", "ridge_prediction",
                    "baseline_prediction"
                ]),
                (performance, [
                    "Ridge_MAE", "Baseline_MAE",
                    "Improvement_mm", "Improvement_percent"
                ]),
            ]:
                for col in numeric_cols:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")

            regions = sorted(forecast_data["region"].dropna().unique().tolist())
            selected_region = st.selectbox(
                "Select study region",
                regions,
                index=regions.index("Punjab and upper Indus")
                if "Punjab and upper Indus" in regions else 0,
            )

            region_perf = performance[
                performance["region"] == selected_region
            ]
            region_validation = validation[
                validation["region"] == selected_region
            ].sort_values("year")
            region_history = forecast_data[
                forecast_data["region"] == selected_region
            ].sort_values("year")

            # ---------------------------------------------------------------
            # FUTURE MONSOON FORECAST: user supplies same-definition inputs
            # ---------------------------------------------------------------
            st.markdown(
                '<div class="fl-card"><div class="fl-card-title">'
                '🛰️ Generate a regional monsoon estimate</div>',
                unsafe_allow_html=True,
            )
            st.write(
                "Upload March–May predictor values for a future monsoon year. "
                "The app will fit the same historical Ridge experiment and "
                "apply the region-specific method selected by the backtest."
            )
            st.info(
                "Required inputs: March–May TerraClimate precipitation total "
                "(pre_rainfall_mm) and mean soil moisture "
                "(pre_soil_moisture_mm), one row per study region. The current "
                "historical training file ends in 2021; it does not contain "
                "2026 or 2027 predictor observations."
            )
            default_forecast_year = max(2027, int(pd.to_numeric(
                forecast_data["year"], errors="coerce").max()) + 1)
            template = make_forecast_template(regions, default_forecast_year)
            st.download_button(
                "Download future-input CSV template",
                data=template.to_csv(index=False).encode("utf-8"),
                file_name=f"FloodLens_Monsoon_Inputs_{default_forecast_year}.csv",
                mime="text/csv",
                key="future_forecast_template",
            )
            forecast_upload = st.file_uploader(
                "Upload March–May regional predictor CSV",
                type=["csv"],
                key="future_monsoon_predictors",
                help=("Required columns: year, region, pre_rainfall_mm, "
                      "pre_soil_moisture_mm. Include all four regions once."),
            )
            input_mode = st.radio(
                "How should these inputs be interpreted?",
                ["Observed March–May data", "Scenario assumptions"],
                horizontal=True,
                key="forecast_input_mode",
            )
            if forecast_upload is not None:
                try:
                    future_inputs = pd.read_csv(forecast_upload)
                    required_future_cols = {
                        "year", "region", "pre_rainfall_mm", "pre_soil_moisture_mm"
                    }
                    missing_future_cols = required_future_cols - set(future_inputs.columns)
                    if missing_future_cols:
                        st.error("Missing input columns: " + ", ".join(sorted(missing_future_cols)))
                    else:
                        future_inputs = future_inputs[list(required_future_cols)].copy()
                        future_inputs["year"] = pd.to_numeric(future_inputs["year"], errors="coerce")
                        for col in ["pre_rainfall_mm", "pre_soil_moisture_mm"]:
                            future_inputs[col] = pd.to_numeric(future_inputs[col], errors="coerce")
                        future_inputs["region"] = future_inputs["region"].astype(str).str.strip()
                        history_end = int(pd.to_numeric(forecast_data["year"], errors="coerce").max())
                        input_years = future_inputs["year"].dropna().unique()
                        input_regions = future_inputs["region"].tolist()
                        valid_regions = set(regions)
                        problems = []
                        if len(input_years) != 1:
                            problems.append("Use one common forecast year in all rows.")
                        elif input_years[0] <= history_end:
                            problems.append(f"Forecast year must be after the training period ending {history_end}.")
                        if len(input_regions) != len(regions) or set(input_regions) != valid_regions:
                            problems.append("Include each of the four study regions exactly once, with exact region names.")
                        if future_inputs["region"].duplicated().any():
                            problems.append("Duplicate region rows were found.")
                        if future_inputs[["pre_rainfall_mm", "pre_soil_moisture_mm"]].isna().any().any():
                            problems.append("Predictor values must be numeric and non-empty.")
                        if (future_inputs[["pre_rainfall_mm", "pre_soil_moisture_mm"]] < 0).any().any():
                            problems.append("Predictor values cannot be negative.")
                        if problems:
                            for problem in problems:
                                st.error(problem)
                        else:
                            forecast_result = run_regional_forecast(
                                forecast_data, performance, future_inputs
                            )
                            forecast_year = int(input_years[0])
                            mode_note = (
                                "Scenario projection only — inputs are assumptions, not observed measurements."
                                if input_mode == "Scenario assumptions"
                                else "Model estimate based on user-supplied observed March–May predictors."
                            )
                            st.success(f"Regional monsoon estimates generated for {forecast_year}.")
                            st.caption(mode_note)
                            selected_forecast = forecast_result[
                                forecast_result["region"] == selected_region
                            ]
                            if not selected_forecast.empty:
                                selected_row = selected_forecast.iloc[0]
                                f1, f2, f3 = st.columns(3)
                                f1.metric("Selected regional estimate", f'{selected_row["selected_prediction_mm"]:,.1f} mm')
                                f2.metric("Ridge estimate", f'{selected_row["ridge_prediction_mm"]:,.1f} mm')
                                f3.metric("Historical baseline", f'{selected_row["baseline_prediction_mm"]:,.1f} mm')
                                st.caption(
                                    f"Selected method: {selected_row['recommended_method']} · "
                                    "Target: June–September total rainfall"
                                )
                            display_forecast = forecast_result[[
                                "year", "region", "pre_rainfall_mm", "pre_soil_moisture_mm",
                                "ridge_prediction_mm", "baseline_prediction_mm",
                                "recommended_method", "selected_prediction_mm"
                            ]].sort_values("region")
                            st.dataframe(display_forecast.round(2), use_container_width=True, hide_index=True)
                            fig_future = px.bar(
                                display_forecast,
                                x="region", y="selected_prediction_mm",
                                color="recommended_method", text_auto=".1f",
                                title=f"{forecast_year} experimental regional monsoon estimates",
                                labels={"region": "Study region", "selected_prediction_mm": "Estimated rainfall (mm)"},
                            )
                            st.plotly_chart(style_fig(fig_future, height=390), use_container_width=True)
                            st.download_button(
                                "Download regional estimates",
                                data=display_forecast.to_csv(index=False).encode("utf-8"),
                                file_name=f"FloodLens_Regional_Monsoon_Estimates_{forecast_year}.csv",
                                mime="text/csv",
                                key="download_generated_forecast",
                            )
                            st.warning(
                                "Experimental estimate only. The model has five held-out "
                                "validation years per region, limited overall improvement, "
                                "and no calibrated prediction interval. It is not a flood warning."
                            )
                except Exception as exc:
                    st.error(f"Could not process the predictor file: {exc}")
            else:
                st.caption(
                    "No forecast has been generated yet. Download the template, "
                    "fill it with real March–May regional predictor values, and upload it here."
                )
            st.markdown("</div>", unsafe_allow_html=True)

            if region_perf.empty:
                st.warning("No performance summary is available for this region.")
            else:
                perf_row = region_perf.iloc[0]
                method = str(perf_row["Recommended_method"])
                method_label = (
                    "Experimental Ridge"
                    if "Ridge" in method
                    else "Historical baseline"
                )

                st.markdown(
                    '<div class="fl-card"><div class="fl-card-title">'
                    'Regional validation summary</div>',
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Selected method", method_label)
                m2.metric("Recommended method MAE",
                          f'{perf_row["Ridge_MAE"] if "Ridge" in method else perf_row["Baseline_MAE"]:.2f} mm')
                m3.metric("Ridge MAE", f'{perf_row["Ridge_MAE"]:.2f} mm')
                m4.metric("Baseline MAE", f'{perf_row["Baseline_MAE"]:.2f} mm')
                st.markdown("</div>", unsafe_allow_html=True)

                st.caption(
                    f"Backtest period: 2017–2021 · "
                    f"MAE improvement versus baseline: "
                    f"{perf_row['Improvement_percent']:.2f}%"
                )

            st.markdown(
                '<div class="fl-card"><div class="fl-card-title">'
                'Historical monsoon rainfall · 2001–2021</div>',
                unsafe_allow_html=True,
            )
            fig_history = px.line(
                region_history,
                x="year",
                y="monsoon_rainfall_mm",
                markers=True,
                title=f"{selected_region}: observed June–September rainfall",
                labels={
                    "year": "Year",
                    "monsoon_rainfall_mm": "Monsoon rainfall (mm)",
                },
            )
            fig_history.update_traces(
                line=dict(color=ACCENT_CYAN, width=2.2),
                marker=dict(size=7),
            )
            st.plotly_chart(
                style_fig(fig_history, height=390),
                use_container_width=True,
            )
            hud_caption("HISTORICAL GPM IMERG REGIONAL MONSOON TOTALS")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                '<div class="fl-card"><div class="fl-card-title">'
                'Chronological validation · 2017–2021</div>',
                unsafe_allow_html=True,
            )
            if not region_validation.empty:
                plot_validation = region_validation.melt(
                    id_vars=["year", "actual"],
                    value_vars=["ridge_prediction", "baseline_prediction"],
                    var_name="method",
                    value_name="predicted",
                )
                plot_validation["method"] = plot_validation["method"].map({
                    "ridge_prediction": "Ridge",
                    "baseline_prediction": "Historical baseline",
                })
                fig_validation = go.Figure()
                fig_validation.add_trace(go.Scatter(
                    x=region_validation["year"],
                    y=region_validation["actual"],
                    mode="lines+markers",
                    name="Observed GPM",
                    line=dict(color="#FFFFFF", width=2.5),
                    marker=dict(size=8),
                ))
                for method_name, color in [
                    ("Ridge", ACCENT_CYAN),
                    ("Historical baseline", ACCENT_ORANGE),
                ]:
                    method_rows = plot_validation[
                        plot_validation["method"] == method_name
                    ]
                    fig_validation.add_trace(go.Scatter(
                        x=method_rows["year"],
                        y=method_rows["predicted"],
                        mode="lines+markers",
                        name=method_name,
                        line=dict(color=color, width=2, dash="dash"),
                        marker=dict(size=7),
                    ))
                fig_validation.update_layout(
                    xaxis_title="Test year",
                    yaxis_title="June–September rainfall (mm)",
                )
                st.plotly_chart(
                    style_fig(fig_validation, height=420),
                    use_container_width=True,
                )
                st.dataframe(
                    region_validation[
                        ["year", "actual", "ridge_prediction", "baseline_prediction"]
                    ].round(2),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning("No chronological validation rows were found for this region.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                '<div class="fl-card"><div class="fl-card-title">'
                'Regional model comparison</div>',
                unsafe_allow_html=True,
            )
            comparison = performance.copy()
            comparison = comparison.sort_values("Ridge_MAE")
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=comparison["region"],
                y=comparison["Ridge_MAE"],
                name="Ridge MAE",
                marker_color=ACCENT_CYAN,
            ))
            fig_compare.add_trace(go.Bar(
                x=comparison["region"],
                y=comparison["Baseline_MAE"],
                name="Historical baseline MAE",
                marker_color=ACCENT_ORANGE,
            ))
            fig_compare.update_layout(
                barmode="group",
                xaxis_title="Study region",
                yaxis_title="Mean absolute error (mm)",
            )
            st.plotly_chart(
                style_fig(fig_compare, height=420),
                use_container_width=True,
            )
            st.dataframe(
                comparison[
                    ["region", "Ridge_MAE", "Baseline_MAE",
                     "Improvement_percent", "Recommended_method"]
                ].round(2),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.warning(
                "Interpretation: validation covers only five held-out years "
                "(2017–2021). Regional performance is preliminary. The model "
                "must not be interpreted as an operational flood-warning system."
            )

            with st.expander("Inspect the model input dataset"):
                st.dataframe(
                    forecast_data[forecast_data["region"] == selected_region],
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download regional forecast dataset",
                data=forecast_data.to_csv(index=False).encode("utf-8"),
                file_name="FloodLens_Regional_Forecast_Dataset.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# PAGE: About & data
# ---------------------------------------------------------------------------
elif page == "About & data":
    st.markdown('<div class="fl-section">Mission Briefing</div>', unsafe_allow_html=True)
    st.markdown("### 📘 About FloodLens")

    st.markdown('<div class="fl-card"><div class="fl-card-title">Overview</div>', unsafe_allow_html=True)
    st.write(
        "FloodLens is an exploratory Earth-system trend detection platform. "
        "Its initial Pakistan case study uses NASA GPM IMERG monthly "
        "precipitation data (2001–2024), with SRTM elevation as static "
        "geographical context."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="fl-card"><div class="fl-card-title">Scientific Interpretation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - Rainfall trends are estimates, not flood forecasts.
        - The national aggregate trend and local pixel trends are different analyses.
        - The original Mann–Kendall pixel tests found no grid cells significant
          after Benjamini–Hochberg false-discovery-rate correction.
        - Elevation is static context, not a time-varying trend variable.
        - Satellite/model rainfall estimates have uncertainty and spatial resolution limits.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="fl-card"><div class="fl-card-title">Data Files Expected in data/</div>', unsafe_allow_html=True)
    st.code(
        "Pakistan_GPM_Rainfall_2001_2024.csv\n"
        "FloodLens_Pakistan_Pixel_Trends_2001_2024.csv\n"
        "pakistan_boundary.geojson\n"
        "FloodLens_Pakistan_Terrain_1km.tif  # optional\\n"
        "FloodLens_Regional_Forecast_Dataset.csv\\n"
        "FloodLens_Forecast_Validation.csv\\n"
        "FloodLens_Final_Regional_Performance.csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="fl-card"><div class="fl-card-title">Data Sources</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <span class="fl-badge">🛰️ NASA GPM IMERG Monthly V07</span>
        <span class="fl-badge orange">🏔️ USGS SRTMGL1</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Primary rainfall source: NASA GPM IMERG Monthly V07. "
        "Terrain source: USGS SRTMGL1."
    )
