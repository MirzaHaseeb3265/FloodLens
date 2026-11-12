import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

try:
    import rasterio
except ImportError:
    rasterio = None

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FloodLens AI | Pakistan Earth-System Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"

MONTHLY_FILE = DATA / "Pakistan_GPM_Rainfall_2001_2024.csv"
SPATIAL_FILE = DATA / "FloodLens_Pakistan_Spatial_Rainfall_2001_2024.csv"
TRENDS_FILE = DATA / "FloodLens_Pakistan_Pixel_Trends_2001_2024.csv"
BOUNDARY_FILE = DATA / "pakistan_boundary.geojson"
TERRAIN_FILE = DATA / "FloodLens_Pakistan_Terrain_1km.tif"

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------
BG_MAIN = "#080D18"
BG_SECOND = "#101827"
PANEL = "#141F30"
ACCENT = "#27D9F2"
ACCENT_2 = "#648CFF"
HIGHLIGHT = "#A78BFA"
TEXT = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
POS = "#34D399"
NEG = "#FB7185"
NEUTRAL = "#64748B"

# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

.stApp {{
    background: radial-gradient(ellipse at top, #0d1526 0%, {BG_MAIN} 55%);
    color: {TEXT};
}}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {BG_SECOND};
    border-right: 1px solid rgba(39, 217, 242, 0.08);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

/* Radio nav in sidebar */
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 2px;
    border-left: 2px solid transparent;
    transition: all 0.15s ease;
    color: {TEXT_MUTED};
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: rgba(39, 217, 242, 0.06);
    color: {TEXT};
}}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: rgba(39, 217, 242, 0.09);
    border-left: 2px solid {ACCENT};
    color: {TEXT};
}}

/* Buttons */
.stButton > button {{
    background: {PANEL};
    color: {TEXT};
    border: 1px solid rgba(39, 217, 242, 0.2);
    border-radius: 8px;
}}
.stButton > button:hover {{
    border-color: {ACCENT};
    color: {ACCENT};
}}

/* Metrics (fallback styling) */
[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid rgba(39, 217, 242, 0.12);
    border-radius: 10px;
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] p {{
    color: {TEXT_MUTED} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-family: 'JetBrains Mono', monospace;
}}

/* Expander */
.streamlit-expanderHeader {{
    background: {PANEL};
    border-radius: 8px;
    color: {TEXT_MUTED};
}}

/* Divider */
hr {{ border-color: rgba(148, 163, 184, 0.15); }}

/* Custom classes */
.fl-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0 20px 0;
    border-bottom: 1px solid rgba(39, 217, 242, 0.08);
    margin-bottom: 28px;
    flex-wrap: wrap;
    gap: 16px;
}}
.fl-brand {{
    display: flex; align-items: center; gap: 14px;
}}
.fl-logo {{
    width: 40px; height: 40px;
    border-radius: 10px;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.1rem; color: {BG_MAIN};
    box-shadow: 0 0 24px rgba(39, 217, 242, 0.25);
}}
.fl-title {{
    font-size: 1.15rem; font-weight: 600;
    color: {TEXT}; letter-spacing: -0.01em; line-height: 1.1;
}}
.fl-sub {{
    font-size: 0.75rem; color: {TEXT_MUTED}; letter-spacing: 0.02em;
    margin-top: 2px;
}}
.fl-header-badges {{
    display: flex; gap: 10px; flex-wrap: wrap;
}}
.fl-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: {PANEL};
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 0.72rem;
    color: {TEXT_MUTED};
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}}
.fl-badge-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {POS};
    box-shadow: 0 0 8px {POS};
}}
.fl-badge-dot.accent {{ background: {ACCENT}; box-shadow: 0 0 8px {ACCENT}; }}

.fl-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.18em;
    color: {ACCENT}; text-transform: uppercase;
    margin-bottom: 10px;
}}
.fl-h1 {{
    font-size: 2.4rem; font-weight: 600;
    letter-spacing: -0.025em; line-height: 1.1;
    color: {TEXT}; margin-bottom: 12px;
}}
.fl-h1 .accent {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.fl-lead {{
    color: {TEXT_MUTED}; font-size: 1rem; line-height: 1.6;
    max-width: 780px; margin-bottom: 32px;
}}

.fl-kpi {{
    background: linear-gradient(180deg, {PANEL} 0%, {BG_SECOND} 100%);
    border: 1px solid rgba(39, 217, 242, 0.10);
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
    position: relative;
    overflow: hidden;
}}
.fl-kpi::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, {ACCENT}, transparent);
}}
.fl-kpi-label {{
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: {TEXT_MUTED};
    margin-bottom: 8px; font-family: 'JetBrains Mono', monospace;
}}
.fl-kpi-value {{
    font-size: 1.7rem; font-weight: 600;
    color: {TEXT}; font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em; line-height: 1;
}}
.fl-kpi-value.pos {{ color: {POS}; }}
.fl-kpi-value.neg {{ color: {NEG}; }}
.fl-kpi-value.neutral {{ color: {TEXT}; }}
.fl-kpi-sub {{
    font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 8px;
}}

.fl-section-title {{
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.15em; color: {TEXT_MUTED};
    margin: 28px 0 14px 0; font-family: 'JetBrains Mono', monospace;
    display: flex; align-items: center; gap: 10px;
}}
.fl-section-title::after {{
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(148,163,184,0.2), transparent);
}}

.fl-panel {{
    background: {PANEL};
    border: 1px solid rgba(148, 163, 184, 0.10);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 12px;
}}
.fl-panel-title {{
    font-size: 0.95rem; font-weight: 600; color: {TEXT};
    margin-bottom: 4px;
}}
.fl-panel-sub {{
    font-size: 0.8rem; color: {TEXT_MUTED}; margin-bottom: 16px;
}}

.fl-interpretation {{
    background: linear-gradient(180deg, rgba(39,217,242,0.04) 0%, rgba(20,31,48,0.4) 100%);
    border-left: 3px solid {ACCENT};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 20px 0;
    color: {TEXT_MUTED}; font-size: 0.9rem; line-height: 1.65;
}}
.fl-interpretation strong {{ color: {TEXT}; font-weight: 600; }}

.fl-warning {{
    background: rgba(251, 113, 133, 0.06);
    border-left: 3px solid {NEG};
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
    color: {TEXT_MUTED}; font-size: 0.88rem; line-height: 1.6;
}}
.fl-warning strong {{ color: {NEG}; }}

.fl-stat-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 16px 0;
}}
.fl-stat {{
    background: {BG_SECOND};
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 10px;
    padding: 14px 16px;
}}
.fl-stat-label {{
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: {TEXT_MUTED};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 6px;
}}
.fl-stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem; font-weight: 500; color: {TEXT};
}}
.fl-footer {{
    margin-top: 60px; padding-top: 20px;
    border-top: 1px solid rgba(148,163,184,0.1);
    color: {TEXT_MUTED}; font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PLOTLY TEMPLATE
# ---------------------------------------------------------------------------
floodlens_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=12),
        title=dict(font=dict(size=14, color=TEXT), x=0, xanchor="left"),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.15)",
            linecolor="rgba(148,163,184,0.15)",
            tickfont=dict(color=TEXT_MUTED, size=11),
            title=dict(font=dict(color=TEXT_MUTED, size=11)),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.08)",
            zerolinecolor="rgba(148,163,184,0.15)",
            linecolor="rgba(148,163,184,0.15)",
            tickfont=dict(color=TEXT_MUTED, size=11),
            title=dict(font=dict(color=TEXT_MUTED, size=11)),
        ),
        colorway=[ACCENT, ACCENT_2, HIGHLIGHT, POS, NEG],
        hoverlabel=dict(
            bgcolor=PANEL, bordercolor=ACCENT,
            font=dict(color=TEXT, family="Inter"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148,163,184,0.15)",
            borderwidth=1,
            font=dict(color=TEXT_MUTED, size=11),
        ),
        margin=dict(l=10, r=10, t=45, b=10),
    )
)
pio.templates["floodlens"] = floodlens_template
pio.templates.default = "floodlens"

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_data
def read_boundary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_boundary_lines(fig, geojson, color="#27D9F2", width=1.2):
    if geojson.get("type") == "FeatureCollection":
        geometries = [f.get("geometry") for f in geojson.get("features", []) if f.get("geometry")]
    elif geojson.get("type") == "Feature":
        geometries = [geojson.get("geometry")]
    else:
        geometries = [geojson]

    def add_ring(ring):
        if not ring:
            return
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False,
            name="Pakistan boundary",
        ))

    for geom in geometries:
        if not geom:
            continue
        kind = geom.get("type")
        coords = geom.get("coordinates", [])
        if kind == "Polygon":
            for ring in coords:
                add_ring(ring)
        elif kind == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    add_ring(ring)
        elif kind == "GeometryCollection":
            for sub in geom.get("geometries", []):
                sk = sub.get("type")
                sc = sub.get("coordinates", [])
                if sk == "Polygon":
                    for ring in sc:
                        add_ring(ring)
                elif sk == "MultiPolygon":
                    for poly in sc:
                        for ring in poly:
                            add_ring(ring)


def load_monthly():
    if not MONTHLY_FILE.exists():
        return None
    df = read_csv(MONTHLY_FILE)
    if "date" not in df or "monthly_rainfall_mm" not in df:
        st.error("Monthly CSV needs 'date' and 'monthly_rainfall_mm' columns.")
        return None
    df = df[["date", "monthly_rainfall_mm"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["monthly_rainfall_mm"] = pd.to_numeric(df["monthly_rainfall_mm"], errors="coerce")
    return df.dropna().sort_values("date")


def load_trends():
    if not TRENDS_FILE.exists():
        return None
    df = read_csv(TRENDS_FILE)
    req = {"latitude", "longitude", "sen_slope_mm_year"}
    if not req.issubset(df.columns):
        st.error("Trend CSV must include latitude, longitude, sen_slope_mm_year.")
        return None
    for col in ["latitude", "longitude", "sen_slope_mm_year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    p_col = "adjusted_p" if "adjusted_p" in df.columns else "adjusted_p_value"
    if p_col in df.columns:
        df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
        df["significant_fdr"] = df[p_col] < 0.05
    elif "significant" in df.columns:
        df["significant_fdr"] = df["significant"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["significant_fdr"] = False
    return df.dropna(subset=["latitude", "longitude", "sen_slope_mm_year"])


# ---------------------------------------------------------------------------
# PLOTLY FIGURE BUILDERS
# ---------------------------------------------------------------------------
def kpi(label, value, sub=None, tone="neutral"):
    cls = f"fl-kpi-value {tone}"
    sub_html = f'<div class="fl-kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="fl-kpi">
        <div class="fl-kpi-label">{label}</div>
        <div class="{cls}">{value}</div>
        {sub_html}
    </div>
    """


def annual_figure(annual):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["annual_rainfall_mm"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2.2, shape="spline", smoothing=0.4),
        marker=dict(size=6, color=ACCENT, line=dict(color=BG_MAIN, width=1)),
        fill="tozeroy",
        fillcolor="rgba(39,217,242,0.07)",
        hovertemplate="<b>%{x}</b><br>Annual rainfall: %{y:.1f} mm<extra></extra>",
    ))
    # Simple linear trend overlay
    if len(annual) > 2:
        z = np.polyfit(annual["year"], annual["annual_rainfall_mm"], 1)
        trend_y = np.polyval(z, annual["year"])
        fig.add_trace(go.Scatter(
            x=annual["year"], y=trend_y,
            mode="lines",
            line=dict(color=HIGHLIGHT, width=1.4, dash="dot"),
            name="Linear trend",
            hoverinfo="skip",
        ))
    fig.update_layout(
        height=380,
        title="Annual rainfall total — Pakistan (2001–2024)",
        xaxis_title="Year",
        yaxis_title="Annual rainfall (mm)",
        showlegend=False,
    )
    return fig


def make_trend_map(trends, boundary, mode):
    plot_df = trends.copy()
    if mode == "significance":
        plot_df["status"] = np.where(
            plot_df["significant_fdr"], "Significant (FDR)", "Not significant (FDR)"
        )
        fig = px.scatter(
            plot_df, x="longitude", y="latitude", color="status",
            color_discrete_map={
                "Significant (FDR)": POS,
                "Not significant (FDR)": NEUTRAL,
            },
            hover_data={
                "sen_slope_mm_year": ":.2f",
                "latitude": ":.2f",
                "longitude": ":.2f",
            },
        )
    else:
        # Diverging scale centered at zero, tuned to the theme
        fig = px.scatter(
            plot_df, x="longitude", y="latitude",
            color="sen_slope_mm_year",
            color_continuous_scale=[
                [0.0, NEG],
                [0.35, "#c084a0"],
                [0.5, NEUTRAL],
                [0.65, "#5a8fd4"],
                [1.0, ACCENT],
            ],
            range_color=(-8, 16),
            hover_data={
                "sen_slope_mm_year": ":.2f",
                "latitude": ":.2f",
                "longitude": ":.2f",
                "significant_fdr": True,
            },
        )
    if boundary is not None:
        add_boundary_lines(fig, boundary, color="#648CFF", width=1.0)
    fig.update_traces(marker=dict(size=5.5, opacity=0.9, line=dict(width=0)))
    fig.update_layout(
        height=680,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=[60, 78], title="Longitude", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[22, 38], title="Latitude"),
        legend=dict(orientation="h", y=1.03, x=0),
        coloraxis_colorbar=dict(
            title=dict(text="mm/yr", font=dict(size=11)),
            thickness=10, len=0.6,
            tickfont=dict(size=10),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# HEADER + SIDEBAR
# ---------------------------------------------------------------------------
def render_header():
    st.markdown(f"""
    <div class="fl-header">
        <div class="fl-brand">
            <div class="fl-logo">FL</div>
            <div>
                <div class="fl-title">FloodLens AI</div>
                <div class="fl-sub">Pakistan Earth-System Intelligence Platform</div>
            </div>
        </div>
        <div class="fl-header-badges">
            <div class="fl-badge">
                <span class="fl-badge-dot accent"></span>
                NASA GPM IMERG · SRTM
            </div>
            <div class="fl-badge">2001–2024</div>
            <div class="fl-badge">
                <span class="fl-badge-dot"></span>
                Analysis engine ready
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                    letter-spacing:0.16em; color:{ACCENT}; text-transform:uppercase;
                    margin-bottom:14px;">Control Panel</div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Section",
            [
                "01 · National Overview",
                "02 · Rainfall Explorer",
                "03 · Spatial Trend Intelligence",
                "04 · Terrain Explorer",
                "05 · Methodology & Data",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="border-top:1px solid rgba(148,163,184,0.12); padding-top:16px;
                    font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                    color:{TEXT_MUTED}; line-height:1.9;">
            <div>STUDY AREA &nbsp;·&nbsp; Pakistan</div>
            <div>PERIOD &nbsp;·&nbsp; 2001 – 2024</div>
            <div>RAINFALL &nbsp;·&nbsp; GPM IMERG V07</div>
            <div>TERRAIN &nbsp;·&nbsp; SRTMGL1</div>
        </div>
        """, unsafe_allow_html=True)
    return page


def render_footer():
    st.markdown(f"""
    <div class="fl-footer">
        <div>FloodLens AI · Earth-System Intelligence</div>
        <div>Data: NASA GPM IMERG · USGS SRTM</div>
        <div>Static historical analysis · not a forecast</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
monthly = load_monthly()
trends = load_trends()
boundary = read_boundary(BOUNDARY_FILE) if BOUNDARY_FILE.exists() else None

# ---------------------------------------------------------------------------
# SHELL
# ---------------------------------------------------------------------------
render_header()
page = render_sidebar()


# ===========================================================================
# PAGE 1 — NATIONAL OVERVIEW
# ===========================================================================
if page.startswith("01"):
    st.markdown('<div class="fl-eyebrow">National Overview · 01</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fl-h1">Understanding Pakistan\'s '
        '<span class="accent">Changing Earth System</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fl-lead">A 24-year retrospective of rainfall behaviour and terrain context '
        'across Pakistan, derived from NASA GPM IMERG monthly precipitation and SRTM elevation. '
        'This dashboard surfaces spatial and temporal trends — it is not a flood forecast.</div>',
        unsafe_allow_html=True,
    )

    # --- KPI row ---
    n_obs = len(monthly) if monthly is not None else 0
    n_cells = len(trends) if trends is not None else 0
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi("Analysis period", "2001–2024", "24 hydrological years"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi("Monthly observations", f"{n_obs:,}" if n_obs else "—", "National aggregate"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi("Spatial grid cells", f"{n_cells:,}" if n_cells else "—", "Pixel-level trend tests"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("National trend", "Increasing", "Mann–Kendall · Hamed–Rao", tone="pos"), unsafe_allow_html=True)

    # --- National Mann-Kendall result ---
    st.markdown('<div class="fl-section-title">National Statistical Result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="fl-stat-row">
        <div class="fl-stat">
            <div class="fl-stat-label">Test</div>
            <div class="fl-stat-value">Mann–Kendall</div>
        </div>
        <div class="fl-stat">
            <div class="fl-stat-label">Trend direction</div>
            <div class="fl-stat-value" style="color:{POS}">Increasing</div>
        </div>
        <div class="fl-stat">
            <div class="fl-stat-label">Sen's slope</div>
            <div class="fl-stat-value">+5.231 mm / yr</div>
        </div>
        <div class="fl-stat">
            <div class="fl-stat-label">Hamed–Rao p-value</div>
            <div class="fl-stat-value">0.035</div>
        </div>
        <div class="fl-stat">
            <div class="fl-stat-label">Significance (α = 0.05)</div>
            <div class="fl-stat-value" style="color:{POS}">Pass</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Annual rainfall chart ---
    if monthly is not None:
        m = monthly.copy()
        m["year"] = m["date"].dt.year
        annual = (
            m.groupby("year", as_index=False)["monthly_rainfall_mm"].sum()
            .rename(columns={"monthly_rainfall_mm": "annual_rainfall_mm"})
        )
        st.markdown('<div class="fl-section-title">Annual Rainfall Trajectory</div>', unsafe_allow_html=True)
        st.plotly_chart(annual_figure(annual), use_container_width=True)

    # --- Spatial preview ---
    if trends is not None:
        st.markdown('<div class="fl-section-title">Spatial Trend Preview</div>', unsafe_allow_html=True)
        preview = trends.sample(min(3000, len(trends)), random_state=42) if len(trends) > 3000 else trends
        st.plotly_chart(make_trend_map(preview, boundary, "slope"), use_container_width=True)

    # --- Interpretation ---
    st.markdown('<div class="fl-section-title">Scientific Interpretation</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="fl-interpretation">
        <strong>National aggregate signal.</strong> The national mean monthly rainfall series shows a
        statistically significant increasing trend (Mann–Kendall with Hamed–Rao correction for
        autocorrelation, p = 0.035), with a Sen's slope of approximately +5.2 mm/year. This describes
        the <em>aggregate</em> rainfall regime across the analysed domain — it does not imply that
        every region is wetting, nor does it constitute a flood forecast.
        <br><br>
        <strong>Pixel-level vs national.</strong> Individual grid-cell Mann–Kendall tests were performed
        separately and corrected for multiple comparisons using Benjamini–Hochberg FDR. The national
        aggregate and the pixel-level question are distinct statistical inferences — see the Spatial
        Trend Intelligence tab for the pixel-level outcome.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2 — RAINFALL EXPLORER
# ===========================================================================
elif page.startswith("02"):
    st.markdown('<div class="fl-eyebrow">Rainfall Explorer · 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="fl-h1">Monthly Rainfall Workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fl-lead">Explore Pakistan\'s monthly rainfall record through time, '
        'with anomalies computed against a 2001–2020 climatological baseline.</div>',
        unsafe_allow_html=True,
    )

    if monthly is None:
        st.info("Add the monthly rainfall CSV to `data/` to populate this page.")
    else:
        m = monthly.copy()
        m["year"] = m["date"].dt.year
        m["month"] = m["date"].dt.month
        min_year, max_year = int(m["year"].min()), int(m["year"].max())

        c1, c2 = st.columns([1, 2])
        with c1:
            selected_years = st.slider("Year range", min_year, max_year, (min_year, max_year))
        subset = m[m["year"].between(*selected_years)]

        # Monthly series
        st.markdown('<div class="fl-section-title">Monthly Rainfall Series</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["monthly_rainfall_mm"],
            mode="lines",
            line=dict(color=ACCENT, width=1.4),
            fill="tozeroy", fillcolor="rgba(39,217,242,0.06)",
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.1f} mm<extra></extra>",
        ))
        fig.update_layout(height=360, xaxis_title="Date", yaxis_title="Rainfall (mm)")
        st.plotly_chart(fig, use_container_width=True)

        # Annual aggregation
        st.markdown('<div class="fl-section-title">Annual Aggregation</div>', unsafe_allow_html=True)
        annual = (
            subset.groupby("year", as_index=False)["monthly_rainfall_mm"].sum()
            .rename(columns={"monthly_rainfall_mm": "annual_rainfall_mm"})
        )
        fig_a = go.Figure()
        fig_a.add_trace(go.Bar(
            x=annual["year"], y=annual["annual_rainfall_mm"],
            marker_color=ACCENT_2, marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>%{y:.1f} mm<extra></extra>",
        ))
        fig_a.update_layout(height=340, xaxis_title="Year", yaxis_title="Annual rainfall (mm)")
        st.plotly_chart(fig_a, use_container_width=True)

        # Heatmap
        st.markdown('<div class="fl-section-title">Monthly Distribution Heatmap</div>', unsafe_allow_html=True)
        heat = subset.pivot_table(index="year", columns="month", values="monthly_rainfall_mm", aggfunc="mean")
        fig_h = px.imshow(
            heat, aspect="auto", origin="lower",
            color_continuous_scale=[[0, BG_SECOND], [0.5, ACCENT_2], [1, ACCENT]],
            labels=dict(x="Month", y="Year", color="mm"),
        )
        fig_h.update_layout(height=380, coloraxis_colorbar=dict(thickness=10, len=0.6))
        st.plotly_chart(fig_h, use_container_width=True)

        # Anomaly
        st.markdown('<div class="fl-section-title">Monthly Anomalies · Baseline 2001–2020</div>', unsafe_allow_html=True)
        baseline = m[m["year"].between(2001, 2020)].copy()
        climatology = baseline.groupby("month")["monthly_rainfall_mm"].mean()
        m["anomaly_mm"] = m["monthly_rainfall_mm"] - m["month"].map(climatology)
        anom = m[m["year"].between(*selected_years)]
        colors = np.where(anom["anomaly_mm"] >= 0, POS, NEG)
        fig_an = go.Figure()
        fig_an.add_trace(go.Bar(
            x=anom["date"], y=anom["anomaly_mm"],
            marker_color=colors, marker_line_width=0,
            hovertemplate="<b>%{x|%b %Y}</b><br>Anomaly: %{y:.1f} mm<extra></extra>",
        ))
        fig_an.update_layout(height=340, xaxis_title="Date", yaxis_title="Anomaly (mm)")
        st.plotly_chart(fig_an, use_container_width=True)

        st.markdown(f"""
        <div class="fl-interpretation">
            <strong>Reading the anomaly chart.</strong> Bars above zero indicate months wetter than the
            2001–2020 calendar-month mean; bars below zero are drier. Positive anomalies in individual
            months do not by themselves establish a trend — the national Mann–Kendall result is the
            statistically tested summary.
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 3 — SPATIAL TREND INTELLIGENCE
# ===========================================================================
elif page.startswith("03"):
    st.markdown('<div class="fl-eyebrow">Spatial Trend Intelligence · 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="fl-h1">Pixel-Level Trend Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fl-lead">Sen\'s slope estimates per grid cell, with a separate statistical '
        'significance layer derived from pixel-level Mann–Kendall tests after Benjamini–Hochberg '
        'false-discovery-rate correction.</div>',
        unsafe_allow_html=True,
    )

    if trends is None:
        st.info("Add `FloodLens_Pakistan_Pixel_Trends_2001_2024.csv` to `data/`.")
    else:
        # --- The critical statistics callout ---
        n_total = len(trends)
        n_sig = int(trends["significant_fdr"].sum())
        n_not = n_total - n_sig

        st.markdown('<div class="fl-section-title">Multiple-Testing Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="fl-stat-row">
            <div class="fl-stat">
                <div class="fl-stat-label">Total pixels tested</div>
                <div class="fl-stat-value">{n_total:,}</div>
            </div>
            <div class="fl-stat">
                <div class="fl-stat-label">Significant (FDR &lt; 0.05)</div>
                <div class="fl-stat-value" style="color:{NEG}">{n_sig:,}</div>
            </div>
            <div class="fl-stat">
                <div class="fl-stat-label">Not significant</div>
                <div class="fl-stat-value" style="color:{NEUTRAL}">{n_not:,}</div>
            </div>
            <div class="fl-stat">
                <div class="fl-stat-label">Correction method</div>
                <div class="fl-stat-value">Benjamini–Hochberg</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if n_sig == 0:
            st.markdown(f"""
            <div class="fl-warning">
                <strong>No pixel passed the FDR significance threshold.</strong>
                After Benjamini–Hochberg correction for {n_total:,} simultaneous tests, zero pixels
                reached FDR-adjusted p &lt; 0.05. The coloured Sen's slope map therefore shows
                <em>estimated</em> local trends, but none are statistically distinguishable from
                noise at the pixel scale. Individual pixels must not be described as "decreasing"
                or "increasing rainfall" on the basis of this map — only as slope estimates with
                wide uncertainty.
            </div>
            """, unsafe_allow_html=True)

        mode = st.radio(
            "Map layer",
            ["Sen's slope (mm/year)", "Statistical significance"],
            horizontal=True,
            label_visibility="collapsed",
        )
        map_mode = "significance" if "significance" in mode.lower() else "slope"
        st.plotly_chart(make_trend_map(trends, boundary, map_mode), use_container_width=True)

        st.markdown(f"""
        <div class="fl-interpretation">
            <strong>National vs pixel-level inference.</strong> The national aggregate trend
            (Mann–Kendall p = 0.035, Sen's slope +5.231 mm/yr) and the pixel-level tests are
            different statistical questions. The national result describes the trend of the
            spatially averaged series; the pixel-level tests ask whether individual grid cells
            exhibit detectable trends once multiple comparisons are accounted for. It is
            entirely possible — and in this case observed — for the national aggregate to be
            significant while no individual pixel survives FDR correction.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Inspect nearest grid cell"):
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude", 24.0, 37.0, 30.0, 0.1)
            lon = c2.number_input("Longitude", 61.0, 76.0, 70.0, 0.1)
            idx = ((trends["latitude"] - lat) ** 2 + (trends["longitude"] - lon) ** 2).idxmin()
            pt = trends.loc[idx]
            st.markdown(f"""
            <div class="fl-stat-row">
                <div class="fl-stat">
                    <div class="fl-stat-label">Nearest cell</div>
                    <div class="fl-stat-value">{pt['latitude']:.2f}°N, {pt['longitude']:.2f}°E</div>
                </div>
                <div class="fl-stat">
                    <div class="fl-stat-label">Sen's slope</div>
                    <div class="fl-stat-value">{pt['sen_slope_mm_year']:.2f} mm/yr</div>
                </div>
                <div class="fl-stat">
                    <div class="fl-stat-label">FDR-adjusted p</div>
                    <div class="fl-stat-value">{pt.get('adjusted_p', pt.get('adjusted_p_value', float('nan'))):.4f}</div>
                </div>
                <div class="fl-stat">
                    <div class="fl-stat-label">Significant (FDR)</div>
                    <div class="fl-stat-value" style="color:{POS if pt['significant_fdr'] else NEUTRAL}">
                        {"Yes" if pt['significant_fdr'] else "No"}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 4 — TERRAIN EXPLORER
# ===========================================================================
elif page.startswith("04"):
    st.markdown('<div class="fl-eyebrow">Terrain Explorer · 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="fl-h1">Pakistan Terrain Context</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fl-lead">Static elevation context from SRTM 1-arc-second data. '
        'Elevation is not a time-varying variable in this analysis — it is provided as '
        'geographic background for interpreting rainfall behaviour.</div>',
        unsafe_allow_html=True,
    )

    if not TERRAIN_FILE.exists():
        st.info("Optional terrain file not found. Add `FloodLens_Pakistan_Terrain_1km.tif` to `data/`.")
    elif rasterio is None:
        st.error("Install rasterio: `pip install rasterio`")
    else:
        with rasterio.open(TERRAIN_FILE) as src:
            elevation = src.read(1).astype("float32")
            nodata = src.nodata
            bounds = src.bounds
        if nodata is not None:
            elevation[elevation == nodata] = np.nan
        step = max(1, int(max(elevation.shape) / 140))
        small = elevation[::step, ::step]

        emin, emax = np.nanmin(elevation), np.nanmax(elevation)
        st.markdown(f"""
        <div class="fl-stat-row">
            <div class="fl-stat">
                <div class="fl-stat-label">Elevation minimum</div>
                <div class="fl-stat-value">{emin:.0f} m</div>
            </div>
            <div class="fl-stat">
                <div class="fl-stat-label">Elevation maximum</div>
                <div class="fl-stat-value">{emax:.0f} m</div>
            </div>
            <div class="fl-stat">
                <div class="fl-stat-label">Source</div>
                <div class="fl-stat-value">SRTMGL1 · 1 km</div>
            </div>
            <div class="fl-stat">
                <div class="fl-stat-label">Variable type</div>
                <div class="fl-stat-value">Static context</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="fl-section-title">2D Elevation Model</div>', unsafe_allow_html=True)
        fig = px.imshow(
            small, origin="upper", aspect="auto",
            color_continuous_scale=[
                [0.0, "#0a1628"],
                [0.15, "#1e3a5f"],
                [0.35, "#4a7c59"],
                [0.6, "#a8a06a"],
                [0.8, "#c9a875"],
                [1.0, "#f0f0f0"],
            ],
        )
        fig.update_layout(
            height=560,
            coloraxis_colorbar=dict(title=dict(text="Elevation (m)", font=dict(size=11)), thickness=10, len=0.6),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="fl-section-title">3D Terrain Surface</div>', unsafe_allow_html=True)
        show3d = st.checkbox("Render interactive 3D surface", value=False)
        if show3d:
            rows, cols = small.shape
            x = np.linspace(bounds.left, bounds.right, cols)
            y = np.linspace(bounds.top, bounds.bottom, rows)
            X, Y = np.meshgrid(x, y)
            fig3d = go.Figure(go.Surface(
                x=X, y=Y, z=small,
                colorscale=[[0, "#0a1628"], [0.35, "#4a7c59"], [0.6, "#a8a06a"], [1, "#f0f0f0"]],
                showscale=False,
            ))
            fig3d.update_layout(
                height=680,
                scene=dict(
                    xaxis_title="Easting (m)",
                    yaxis_title="Northing (m)",
                    zaxis_title="Elevation (m)",
                    xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.1)"),
                    yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.1)"),
                    zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(148,163,184,0.1)"),
                    aspectratio=dict(x=1, y=1, z=0.3),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig3d, use_container_width=True)


# ===========================================================================
# PAGE 5 — METHODOLOGY & DATA
# ===========================================================================
else:
    st.markdown('<div class="fl-eyebrow">Methodology & Data · 05</div>', unsafe_allow_html=True)
    st.markdown('<div class="fl-h1">Scientific Methodology</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fl-lead">Data sources, statistical tests, and the limits of interpretation '
        'for FloodLens AI.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fl-section-title">Data Sources</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="fl-panel">
        <div class="fl-panel-title">NASA GPM IMERG Monthly V07</div>
        <div class="fl-panel-sub">Global Precipitation Measurement · Integrated Multi-satellitE Retrievals</div>
        Gridded monthly precipitation estimates at 0.1° resolution, aggregated to a national
        monthly series for Pakistan, 2001–2024.
    </div>
    <div class="fl-panel" style="margin-top:12px;">
        <div class="fl-panel-title">USGS SRTMGL1</div>
        <div class="fl-panel-sub">Shuttle Radar Topography Mission · 1 arc-second</div>
        Static elevation context used solely for geographic interpretation — not a trend variable.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fl-section-title">Statistical Methods</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="fl-panel">
        <div class="fl-panel-title">Mann–Kendall trend test</div>
        <div class="fl-panel-sub">Non-parametric monotonic trend detection</div>
        Applied to the national monthly rainfall aggregate with Hamed–Rao correction for
        autocorrelation. Result: <b style="color:{POS}">increasing</b>, p = 0.035.
    </div>
    <div class="fl-panel" style="margin-top:12px;">
        <div class="fl-panel-title">Sen's slope estimator</div>
        <div class="fl-panel-sub">Robust median of pairwise slopes</div>
        National Sen's slope: <b>+5.231 mm/year</b>. Pixel-level Sen's slopes are computed
        per grid cell and shown on the Spatial Trend Intelligence page.
    </div>
    <div class="fl-panel" style="margin-top:12px;">
        <div class="fl-panel-title">Benjamini–Hochberg FDR correction</div>
        <div class="fl-panel-sub">Multiple-testing control at α = 0.05</div>
        Applied to the {len(trends) if trends is not None else 0:,} pixel-level Mann–Kendall tests.
        Result: <b style="color:{NEG}">0 pixels significant</b> after FDR correction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fl-section-title">Interpretation Limits</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="fl-warning">
        <strong>What this dashboard does not claim.</strong>
        <ul style="margin:8px 0 0 0; padding-left:20px;">
            <li>Rainfall trends are estimates — not forecasts and not flood predictions.</li>
            <li>The national aggregate trend and pixel-level trends are different statistical questions.</li>
            <li>No pixel passed FDR correction; pixel-level slopes must not be described as significant trends.</li>
            <li>Satellite precipitation products carry retrieval uncertainty and resolution limits.</li>
            <li>SRTM elevation is static context and does not enter any trend test.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fl-section-title">Expected Data Files</div>', unsafe_allow_html=True)
    st.code(
        "data/\n"
        "├── Pakistan_GPM_Rainfall_2001_2024.csv\n"
        "├── FloodLens_Pakistan_Pixel_Trends_2001_2024.csv\n"
        "├── pakistan_boundary.geojson\n"
        "└── FloodLens_Pakistan_Terrain_1km.tif  # optional",
        language="text",
    )


# ---------------------------------------------------------------------------
render_footer()
