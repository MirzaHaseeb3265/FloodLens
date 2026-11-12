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

# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(
    page_title="FloodLens AI · Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"

MONTHLY_FILE = DATA / "Pakistan_GPM_Rainfall_2001_2024.csv"
TRENDS_FILE = DATA / "FloodLens_Pakistan_Pixel_Trends_2001_2024.csv"
BOUNDARY_FILE = DATA / "pakistan_boundary.geojson"
TERRAIN_FILE = DATA / "FloodLens_Pakistan_Terrain_1km.tif"

# ============================================================================
# TOKENS
# ============================================================================
BG        = "#060b14"
BG_2      = "#0b1421"
PANEL     = "#0f1a2b"
PANEL_2   = "#131f33"
BORDER    = "rgba(39,217,242,0.14)"
BORDER_S  = "rgba(148,163,184,0.10)"
CYAN      = "#27D9F2"
BLUE      = "#648CFF"
VIOLET    = "#A78BFA"
TEXT      = "#F1F5F9"
MUTED     = "#7c8ba1"
POS       = "#34D399"
NEG       = "#FB7185"
NEUTRAL   = "#475569"

# ============================================================================
# CSS — MISSION CONTROL SHELL
# ============================================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {BG}; color: {TEXT}; }}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0.6rem 1rem 1rem 1rem !important; max-width: 100% !important; }}
[data-testid="stSidebar"] {{ display: none; }}

/* ---------- TOP STATUS BAR ---------- */
.mc-topbar {{
    display: grid; grid-template-columns: auto 1fr auto;
    align-items: center; gap: 24px;
    background: {PANEL};
    border: 1px solid {BORDER};
    padding: 10px 18px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
}}
.mc-brand {{ display: flex; align-items: center; gap: 12px; }}
.mc-logo {{
    width: 30px; height: 30px; border-radius: 6px;
    background: linear-gradient(135deg, {CYAN}, {BLUE});
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem; color: {BG};
    box-shadow: 0 0 16px rgba(39,217,242,0.35);
}}
.mc-brand-text {{ font-size: 0.95rem; font-weight: 600; color: {TEXT}; letter-spacing: 0.02em; }}
.mc-brand-sub {{ font-size: 0.6rem; color: {MUTED}; letter-spacing: 0.16em; text-transform: uppercase; margin-top: 1px; }}

.mc-telemetry {{
    display: flex; gap: 26px; justify-content: center;
    font-size: 0.65rem; letter-spacing: 0.08em; color: {MUTED};
}}
.mc-telemetry span b {{ color: {CYAN}; font-weight: 500; }}

.mc-badges {{ display: flex; gap: 8px; }}
.mc-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px;
    border: 1px solid {BORDER_S};
    font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: {MUTED}; background: {BG_2};
}}
.mc-dot {{ width: 5px; height: 5px; border-radius: 50%; background: {POS}; box-shadow: 0 0 6px {POS}; }}
.mc-dot.cyan {{ background: {CYAN}; box-shadow: 0 0 6px {CYAN}; }}

/* ---------- HUD CARD — corner brackets ---------- */
.hud {{
    position: relative;
    background: {PANEL};
    border: 1px solid {BORDER_S};
    padding: 14px 16px;
    height: 100%;
}}
.hud::before, .hud::after {{
    content: ""; position: absolute; width: 10px; height: 10px;
    border-color: {CYAN}; border-style: solid;
}}
.hud::before {{ top: -1px; left: -1px; border-width: 2px 0 0 2px; }}
.hud::after  {{ top: -1px; right: -1px; border-width: 2px 2px 0 0; }}
.hud > .hud-bl, .hud > .hud-br {{
    content: ""; position: absolute; width: 10px; height: 10px;
    border-color: {CYAN}; border-style: solid;
}}
.hud > .hud-bl {{ bottom: -1px; left: -1px; border-width: 0 0 2px 2px; }}
.hud > .hud-br {{ bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }}

.hud-head {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 8px; margin-bottom: 10px;
    border-bottom: 1px solid {BORDER_S};
    font-family: 'JetBrains Mono', monospace;
}}
.hud-title {{ font-size: 0.68rem; letter-spacing: 0.16em; color: {CYAN}; text-transform: uppercase; font-weight: 500; }}
.hud-title.blue {{ color: {BLUE}; }}
.hud-title.violet {{ color: {VIOLET}; }}
.hud-meta {{ font-size: 0.6rem; letter-spacing: 0.1em; color: {MUTED}; text-transform: uppercase; }}

/* ---------- KPI STRIP ---------- */
.kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 10px; }}
.kpi {{
    background: {PANEL}; border: 1px solid {BORDER_S};
    padding: 12px 14px;
    position: relative; overflow: hidden;
}}
.kpi::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 2px;
    background: {CYAN}; box-shadow: 0 0 8px {CYAN};
}}
.kpi-label {{ font-size: 0.6rem; letter-spacing: 0.14em; color: {MUTED}; text-transform: uppercase; font-family: 'JetBrains Mono', monospace; }}
.kpi-value {{ font-size: 1.45rem; color: {TEXT}; font-family: 'JetBrains Mono', monospace; font-weight: 500; letter-spacing: -0.02em; margin-top: 4px; }}
.kpi-value.pos {{ color: {POS}; }}
.kpi-value.neg {{ color: {NEG}; }}
.kpi-sub {{ font-size: 0.62rem; color: {MUTED}; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }}

/* ---------- TABULAR DATA STRIP ---------- */
.mc-table {{ width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; }}
.mc-table th {{
    text-align: left; color: {MUTED}; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.58rem;
    padding: 6px 8px; border-bottom: 1px solid {BORDER_S};
}}
.mc-table td {{ padding: 8px 8px; border-bottom: 1px solid rgba(148,163,184,0.05); color: {TEXT}; }}
.mc-table tr:last-child td {{ border-bottom: none; }}
.mc-table td.pos {{ color: {POS}; }}
.mc-table td.neg {{ color: {NEG}; }}
.mc-table td.mono {{ color: {TEXT}; }}

/* ---------- STATUS LIST ---------- */
.status-list {{ display: flex; flex-direction: column; gap: 0; }}
.status-item {{
    display: grid; grid-template-columns: 10px 1fr auto;
    gap: 10px; align-items: center;
    padding: 9px 0; border-bottom: 1px solid rgba(148,163,184,0.06);
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
}}
.status-item:last-child {{ border-bottom: none; }}
.status-dot {{ width: 6px; height: 6px; border-radius: 50%; }}
.status-name {{ color: {MUTED}; letter-spacing: 0.04em; }}
.status-val {{ color: {TEXT}; font-weight: 500; }}

/* ---------- INTERPRETATION STRIP ---------- */
.mc-note {{
    background: linear-gradient(90deg, rgba(39,217,242,0.05) 0%, transparent 100%);
    border-left: 2px solid {CYAN};
    padding: 10px 14px;
    font-size: 0.72rem; line-height: 1.6; color: {MUTED};
    font-family: 'Inter', sans-serif;
}}
.mc-note strong {{ color: {TEXT}; font-weight: 600; }}
.mc-warn {{
    background: linear-gradient(90deg, rgba(251,113,133,0.06) 0%, transparent 100%);
    border-left: 2px solid {NEG};
    padding: 10px 14px; font-size: 0.72rem; line-height: 1.6; color: {MUTED};
}}
.mc-warn strong {{ color: {NEG}; }}

/* ---------- PAGE NAV (horizontal tabs) ---------- */
.mc-navwrap {{ display: flex; gap: 0; margin-bottom: 10px; border-bottom: 1px solid {BORDER_S}; }}
.mc-navwrap .stRadio {{ flex: 1; }}
.mc-navwrap [role="radiogroup"] {{ flex-direction: row !important; gap: 0 !important; }}
.mc-navwrap [role="radiogroup"] label {{
    background: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; padding: 8px 18px !important;
    margin-right: 0 !important; margin-bottom: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; color: {MUTED} !important;
}}
.mc-navwrap [role="radiogroup"] label:has(input:checked) {{
    color: {CYAN} !important;
    border-bottom: 2px solid {CYAN} !important;
    background: rgba(39,217,242,0.04) !important;
}}

/* Plotly transparent */
.js-plotly-plot .plotly {{ background: transparent !important; }}

/* Reduce Streamlit chrome */
[data-testid="stHorizontalBlock"] {{ gap: 10px !important; }}
.stPlotlyChart {{ margin: 0 !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================================
# PLOTLY TEMPLATE
# ============================================================================
pio.templates["mc"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=MUTED, size=10),
    title=dict(font=dict(size=11, color=TEXT), x=0.02, y=0.95),
    xaxis=dict(gridcolor="rgba(148,163,184,0.06)", zerolinecolor="rgba(148,163,184,0.1)",
               linecolor="rgba(148,163,184,0.1)", tickfont=dict(size=9)),
    yaxis=dict(gridcolor="rgba(148,163,184,0.06)", zerolinecolor="rgba(148,163,184,0.1)",
               linecolor="rgba(148,163,184,0.1)", tickfont=dict(size=9)),
    colorway=[CYAN, BLUE, VIOLET, POS, NEG],
    hoverlabel=dict(bgcolor=PANEL_2, bordercolor=CYAN, font=dict(color=TEXT, family="JetBrains Mono")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=9)),
    margin=dict(l=30, r=10, t=30, b=30),
))
pio.templates.default = "mc"

# ============================================================================
# DATA
# ============================================================================
@st.cache_data
def read_csv(p): return pd.read_csv(p)

@st.cache_data
def read_geojson(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def load_monthly():
    if not MONTHLY_FILE.exists(): return None
    df = read_csv(MONTHLY_FILE)
    if "date" not in df or "monthly_rainfall_mm" not in df: return None
    df = df[["date", "monthly_rainfall_mm"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["monthly_rainfall_mm"] = pd.to_numeric(df["monthly_rainfall_mm"], errors="coerce")
    return df.dropna().sort_values("date")

def load_trends():
    if not TRENDS_FILE.exists(): return None
    df = read_csv(TRENDS_FILE)
    req = {"latitude", "longitude", "sen_slope_mm_year"}
    if not req.issubset(df.columns): return None
    for c in ["latitude", "longitude", "sen_slope_mm_year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    pcol = "adjusted_p" if "adjusted_p" in df.columns else "adjusted_p_value"
    if pcol in df.columns:
        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
        df["significant_fdr"] = df[pcol] < 0.05
    else:
        df["significant_fdr"] = False
    return df.dropna(subset=["latitude", "longitude", "sen_slope_mm_year"])

def add_boundary(fig, gj, color=BLUE, width=0.8):
    if not gj: return
    feats = gj.get("features", []) if gj.get("type") == "FeatureCollection" else [{"geometry": gj}]
    for f in feats:
        geom = f.get("geometry")
        if not geom: continue
        kind = geom.get("type"); coords = geom.get("coordinates", [])
        polys = [coords] if kind == "Polygon" else coords if kind == "MultiPolygon" else []
        for poly in polys:
            for ring in poly:
                xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=width),
                    hoverinfo="skip", showlegend=False,
                ))

# ============================================================================
# COMPONENTS
# ============================================================================
def hud_open(title, meta="", accent="cyan"):
    cls = {"cyan": "", "blue": " blue", "violet": " violet"}.get(accent, "")
    st.markdown(f"""
    <div class="hud">
        <span class="hud-bl"></span><span class="hud-br"></span>
        <div class="hud-head">
            <div class="hud-title{cls}">{title}</div>
            <div class="hud-meta">{meta}</div>
        </div>
    """, unsafe_allow_html=True)

def hud_close():
    st.markdown("</div>", unsafe_allow_html=True)

def topbar():
    st.markdown(f"""
    <div class="mc-topbar">
        <div class="mc-brand">
            <div class="mc-logo">FL</div>
            <div>
                <div class="mc-brand-text">FLOODLENS AI</div>
                <div class="mc-brand-sub">Earth-System Intelligence · Pakistan</div>
            </div>
        </div>
        <div class="mc-telemetry">
            <span>MODE <b>ANALYSIS</b></span>
            <span>PERIOD <b>2001-2024</b></span>
            <span>OBS <b>288 MO</b></span>
            <span>GRID <b>7335 CELLS</b></span>
        </div>
        <div class="mc-badges">
            <div class="mc-chip"><span class="mc-dot cyan"></span>GPM IMERG</div>
            <div class="mc-chip"><span class="mc-dot"></span>SRTM V3</div>
            <div class="mc-chip">v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def nav():
    st.markdown('<div class="mc-navwrap">', unsafe_allow_html=True)
    page = st.radio(
        "nav", ["OVERVIEW", "RAINFALL", "SPATIAL TREND", "TERRAIN", "METHODOLOGY"],
        horizontal=True, label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return page

# ============================================================================
# LOAD
# ============================================================================
monthly = load_monthly()
trends = load_trends()
boundary = read_geojson(BOUNDARY_FILE) if BOUNDARY_FILE.exists() else None

topbar()
page = nav()


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "OVERVIEW":
    n_obs = len(monthly) if monthly is not None else 0
    n_cells = len(trends) if trends is not None else 0
    n_sig = int(trends["significant_fdr"].sum()) if trends is not None else 0

    # ---- KPI strip ----
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Analysis Period</div>
            <div class="kpi-value">2001–2024</div>
            <div class="kpi-sub">24 hydrological years</div></div>
        <div class="kpi"><div class="kpi-label">Monthly Obs</div>
            <div class="kpi-value">{n_obs:,}</div>
            <div class="kpi-sub">national aggregate</div></div>
        <div class="kpi"><div class="kpi-label">Grid Cells</div>
            <div class="kpi-value">{n_cells:,}</div>
            <div class="kpi-sub">pixel-level tests</div></div>
        <div class="kpi"><div class="kpi-label">National Trend</div>
            <div class="kpi-value pos">INCREASING</div>
            <div class="kpi-sub">p = 0.035</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ---- MAIN 3-COLUMN GRID ----
    col_left, col_center, col_right = st.columns([1.05, 2.0, 1.05], gap="small")

    # ---------- LEFT: result table + status ----------
    with col_left:
        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title">National Mann–Kendall</div>
                <div class="hud-meta">SIGNAL</div>
            </div>
            <table class="mc-table">
                <tr><th>Param</th><th>Value</th></tr>
                <tr><td>Direction</td><td class="pos">INCREASING</td></tr>
                <tr><td>Sen's Slope</td><td class="mono">+5.231 mm/yr</td></tr>
                <tr><td>Hamed–Rao p</td><td class="mono">0.035</td></tr>
                <tr><td>α threshold</td><td class="mono">0.050</td></tr>
                <tr><td>Significance</td><td class="pos">PASS</td></tr>
                <tr><td>Test Type</td><td class="mono">MK + HR</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title blue">Data Sources</div>
                <div class="hud-meta">2 ACTIVE</div>
            </div>
            <div class="status-list">
                <div class="status-item">
                    <span class="status-dot" style="background:{POS};box-shadow:0 0 6px {POS}"></span>
                    <span class="status-name">GPM IMERG V07</span>
                    <span class="status-val">MONTHLY</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{POS};box-shadow:0 0 6px {POS}"></span>
                    <span class="status-name">SRTMGL1 Elevation</span>
                    <span class="status-val">STATIC</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{NEUTRAL}"></span>
                    <span class="status-name">Boundary</span>
                    <span class="status-val">GEOJSON</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title violet">FDR Correction</div>
                <div class="hud-meta">BH · α=0.05</div>
            </div>
            <div class="status-list">
                <div class="status-item">
                    <span class="status-dot" style="background:{MUTED}"></span>
                    <span class="status-name">Tested</span>
                    <span class="status-val">{n_cells:,}</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{NEG};box-shadow:0 0 6px {NEG}"></span>
                    <span class="status-name">Significant</span>
                    <span class="status-val" style="color:{NEG}">{n_sig:,}</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{NEUTRAL}"></span>
                    <span class="status-name">Not Significant</span>
                    <span class="status-val">{n_cells-n_sig:,}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- CENTER: hero map ----------
    with col_center:
        st.markdown(f"""
        <div class="hud" style="padding-bottom:6px;">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title">Spatial Sen's Slope · Pakistan</div>
                <div class="hud-meta">LON 60–78 · LAT 22–38 · 7335 PX</div>
            </div>
        """, unsafe_allow_html=True)
        if trends is not None:
            fig = px.scatter(
                trends, x="longitude", y="latitude",
                color="sen_slope_mm_year",
                color_continuous_scale=[
                    [0.0, NEG], [0.35, "#c084a0"], [0.5, NEUTRAL],
                    [0.65, BLUE], [1.0, CYAN],
                ],
                range_color=(-8, 16),
                hover_data={"sen_slope_mm_year": ":.2f", "latitude": ":.2f", "longitude": ":.2f"},
            )
            add_boundary(fig, boundary)
            fig.update_traces(marker=dict(size=4, opacity=0.9))
            fig.update_layout(
                height=470,
                xaxis=dict(range=[60, 78], title="", scaleanchor="y", scaleratio=1, showticklabels=True),
                yaxis=dict(range=[22, 38], title=""),
                coloraxis_colorbar=dict(
                    title=dict(text="mm/yr", font=dict(size=9, color=MUTED)),
                    thickness=8, len=0.55, x=1.02,
                    tickfont=dict(size=8),
                    outlinewidth=0,
                ),
                margin=dict(l=30, r=10, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Add pixel trends CSV to data/")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RIGHT: vertical panels ----------
    with col_right:
        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title">Terrain Context</div>
                <div class="hud-meta">SRTM</div>
            </div>
            <div class="status-list">
                <div class="status-item">
                    <span class="status-dot" style="background:{CYAN}"></span>
                    <span class="status-name">Elevation Range</span>
                    <span class="status-val">0–8600 m</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{BLUE}"></span>
                    <span class="status-name">Resolution</span>
                    <span class="status-val">1 arc-sec</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{NEUTRAL}"></span>
                    <span class="status-name">Variable</span>
                    <span class="status-val">STATIC</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title blue">Study Region</div>
                <div class="hud-meta">PAKISTAN</div>
            </div>
            <div class="status-list">
                <div class="status-item">
                    <span class="status-dot" style="background:{CYAN}"></span>
                    <span class="status-name">Bounding Box</span>
                    <span class="status-val">60–78°E</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{CYAN}"></span>
                    <span class="status-name">Latitude Span</span>
                    <span class="status-val">22–38°N</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{CYAN}"></span>
                    <span class="status-name">Grid Cells</span>
                    <span class="status-val">{n_cells:,}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title violet">System Status</div>
                <div class="hud-meta">READY</div>
            </div>
            <div class="status-list">
                <div class="status-item">
                    <span class="status-dot" style="background:{POS};box-shadow:0 0 6px {POS}"></span>
                    <span class="status-name">Analysis Engine</span>
                    <span class="status-val">ONLINE</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{POS};box-shadow:0 0 6px {POS}"></span>
                    <span class="status-name">Data Cache</span>
                    <span class="status-val">LOADED</span>
                </div>
                <div class="status-item">
                    <span class="status-dot" style="background:{VIOLET};box-shadow:0 0 6px {VIOLET}"></span>
                    <span class="status-name">Version</span>
                    <span class="status-val">v1.0</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- BOTTOM ROW: 3 micro-charts ----
    if monthly is not None:
        m = monthly.copy()
        m["year"] = m["date"].dt.year
        m["month"] = m["date"].dt.month
        annual = m.groupby("year", as_index=False)["monthly_rainfall_mm"].sum()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="small")

        with c1:
            st.markdown(f"""
            <div class="hud" style="padding-bottom:0;">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title">Annual Series</div>
                    <div class="hud-meta">mm/yr</div>
                </div>
            """, unsafe_allow_html=True)
            f1 = go.Figure(go.Scatter(
                x=annual["year"], y=annual["monthly_rainfall_mm"],
                mode="lines+markers",
                line=dict(color=CYAN, width=1.6),
                marker=dict(size=3, color=CYAN),
                fill="tozeroy", fillcolor="rgba(39,217,242,0.08)",
                hovertemplate="%{x} · %{y:.0f} mm<extra></extra>",
            ))
            f1.update_layout(height=180, margin=dict(l=30, r=6, t=6, b=20), showlegend=False)
            st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="hud" style="padding-bottom:0;">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title blue">Monthly Heatmap</div>
                    <div class="hud-meta">YEAR × MONTH</div>
                </div>
            """, unsafe_allow_html=True)
            heat = m.pivot_table(index="year", columns="month", values="monthly_rainfall_mm", aggfunc="mean")
            f2 = px.imshow(
                heat, aspect="auto", origin="lower",
                color_continuous_scale=[[0, BG_2], [0.5, BLUE], [1, CYAN]],
            )
            f2.update_layout(
                height=180, margin=dict(l=30, r=6, t=6, b=20),
                coloraxis_showscale=False,
                xaxis=dict(showticklabels=True, tickfont=dict(size=7)),
                yaxis=dict(showticklabels=True, tickfont=dict(size=7)),
            )
            st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="hud" style="padding-bottom:0;">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title violet">Anomaly Signal</div>
                    <div class="hud-meta">VS 2001–2020</div>
                </div>
            """, unsafe_allow_html=True)
            base = m[m["year"].between(2001, 2020)]
            clim = base.groupby("month")["monthly_rainfall_mm"].mean()
            m["anom"] = m["monthly_rainfall_mm"] - m["month"].map(clim)
            colors = np.where(m["anom"] >= 0, POS, NEG)
            f3 = go.Figure(go.Bar(
                x=m["date"], y=m["anom"],
                marker_color=colors, marker_line_width=0,
                hovertemplate="%{x|%b %Y} · %{y:.1f} mm<extra></extra>",
            ))
            f3.update_layout(height=180, margin=dict(l=30, r=6, t=6, b=20), showlegend=False)
            st.plotly_chart(f3, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- Interpretation ----
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hud">
        <span class="hud-bl"></span><span class="hud-br"></span>
        <div class="hud-head">
            <div class="hud-title">Scientific Interpretation</div>
            <div class="hud-meta">READ</div>
        </div>
        <div class="mc-note">
            <strong>National aggregate:</strong> The national mean monthly series exhibits a
            statistically significant increasing trend (Mann–Kendall with Hamed–Rao correction,
            p = 0.035; Sen's slope ≈ +5.23 mm/yr). This describes the <em>aggregate</em> rainfall
            regime across the study domain — not every pixel wets, and this is not a flood forecast.
        </div>
        <div style="height:8px"></div>
        <div class="mc-warn">
            <strong>Pixel-level:</strong> Of {n_cells:,} grid cells tested with pixel-level
            Mann–Kendall, <strong>0 passed</strong> Benjamini–Hochberg FDR correction. Pixel slopes
            are estimates with wide uncertainty — do not describe any individual pixel as an
            increasing or decreasing rainfall trend.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: RAINFALL
# ============================================================================
elif page == "RAINFALL":
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Monthly Records</div>
            <div class="kpi-value">{(len(monthly) if monthly is not None else 0):,}</div>
            <div class="kpi-sub">2001-01 → 2024-12</div></div>
        <div class="kpi"><div class="kpi-label">Mean Annual</div>
            <div class="kpi-value">{("—" if monthly is None else f"{monthly.groupby(monthly['date'].dt.year)['monthly_rainfall_mm'].sum().mean():.0f}")}</div>
            <div class="kpi-sub">mm / year</div></div>
        <div class="kpi"><div class="kpi-label">Wettest Month</div>
            <div class="kpi-value">{("—" if monthly is None else f"{monthly['monthly_rainfall_mm'].max():.0f}")}</div>
            <div class="kpi-sub">mm</div></div>
        <div class="kpi"><div class="kpi-label">Driest Month</div>
            <div class="kpi-value">{("—" if monthly is None else f"{monthly['monthly_rainfall_mm'].min():.1f}")}</div>
            <div class="kpi-sub">mm</div></div>
    </div>
    """, unsafe_allow_html=True)

    if monthly is None:
        st.info("Add monthly CSV to data/.")
    else:
        m = monthly.copy()
        m["year"] = m["date"].dt.year
        m["month"] = m["date"].dt.month

        c1, c2 = st.columns([1, 4], gap="small")
        with c1:
            st.markdown(f"""
            <div class="hud">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title">Filter Range</div>
                    <div class="hud-meta">YEAR</div>
                </div>
            """, unsafe_allow_html=True)
            yr = st.slider("Year", int(m["year"].min()), int(m["year"].max()), (int(m["year"].min()), int(m["year"].max())), label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
        sub = m[m["year"].between(*yr)]

        with c2:
            st.markdown(f"""
            <div class="hud" style="padding-bottom:0;">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title">Monthly Rainfall Signal</div>
                    <div class="hud-meta">{yr[0]} – {yr[1]}</div>
                </div>
            """, unsafe_allow_html=True)
            f = go.Figure(go.Scatter(
                x=sub["date"], y=sub["monthly_rainfall_mm"],
                mode="lines", line=dict(color=CYAN, width=1.2),
                fill="tozeroy", fillcolor="rgba(39,217,242,0.08)",
                hovertemplate="%{x|%b %Y} · %{y:.1f} mm<extra></extra>",
            ))
            f.update_layout(height=260, margin=dict(l=40, r=10, t=10, b=30))
            st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns([1, 1], gap="small")

        with c3:
            st.markdown(f"""
            <div class="hud" style="padding-bottom:0;">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title blue">Annual Aggregation</div>
                    <div class="hud-meta">mm</div>
                </div>
            """, unsafe_allow_html=True)
            ann = sub.groupby("year", as_index=False)["monthly_rainfall_mm"].sum()
            fa = go.Figure(go.Bar(
                x=ann["year"], y=ann["monthly_rainfall_mm"],
                marker_color=BLUE, marker_line_width=0,
                hovertemplate="%{x} · %{y:.0f} mm<extra></extra>",
            ))
            fa.update_layout(height=240, margin=dict(l=40, r=10, t=10, b=30))
            st.plotly_chart(fa, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="hud" style="padding-bottom:0;">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title violet">Anomaly vs 2001–2020</div>
                    <div class="hud-meta">± mm</div>
                </div>
            """, unsafe_allow_html=True)
            base = m[m["year"].between(2001, 2020)]
            clim = base.groupby("month")["monthly_rainfall_mm"].mean()
            sub2 = sub.copy()
            sub2["anom"] = sub2["monthly_rainfall_mm"] - sub2["month"].map(clim)
            colors = np.where(sub2["anom"] >= 0, POS, NEG)
            fan = go.Figure(go.Bar(
                x=sub2["date"], y=sub2["anom"],
                marker_color=colors, marker_line_width=0,
                hovertemplate="%{x|%b %Y} · %{y:.1f} mm<extra></extra>",
            ))
            fan.update_layout(height=240, margin=dict(l=40, r=10, t=10, b=30))
            st.plotly_chart(fan, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# PAGE: SPATIAL
# ============================================================================
elif page == "SPATIAL TREND":
    n_cells = len(trends) if trends is not None else 0
    n_sig = int(trends["significant_fdr"].sum()) if trends is not None else 0

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Pixels Tested</div>
            <div class="kpi-value">{n_cells:,}</div>
            <div class="kpi-sub">MK per cell</div></div>
        <div class="kpi"><div class="kpi-label">Significant (FDR)</div>
            <div class="kpi-value neg">{n_sig:,}</div>
            <div class="kpi-sub">α = 0.05</div></div>
        <div class="kpi"><div class="kpi-label">Not Significant</div>
            <div class="kpi-value">{n_cells - n_sig:,}</div>
            <div class="kpi-sub">FDR-corrected</div></div>
        <div class="kpi"><div class="kpi-label">Correction</div>
            <div class="kpi-value" style="font-size:1rem;">BENJAMINI</div>
            <div class="kpi-sub">HOCHBERG</div></div>
    </div>
    """, unsafe_allow_html=True)

    if trends is None:
        st.info("Add trends CSV to data/.")
    else:
        cL, cR = st.columns([3, 1], gap="small")

        with cL:
            st.markdown(f"""
            <div class="hud" style="padding-bottom:0;">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title">Pakistan Spatial Sen's Slope</div>
                    <div class="hud-meta">7335 CELLS · MM/YR</div>
                </div>
            """, unsafe_allow_html=True)
            fig = px.scatter(
                trends, x="longitude", y="latitude",
                color="sen_slope_mm_year",
                color_continuous_scale=[[0, NEG], [0.35, "#c084a0"], [0.5, NEUTRAL], [0.65, BLUE], [1, CYAN]],
                range_color=(-8, 16),
                hover_data={"sen_slope_mm_year": ":.2f", "latitude": ":.2f", "longitude": ":.2f"},
            )
            add_boundary(fig, boundary)
            fig.update_traces(marker=dict(size=5, opacity=0.9))
            fig.update_layout(
                height=640,
                xaxis=dict(range=[60, 78], scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[22, 38]),
                coloraxis_colorbar=dict(title=dict(text="mm/yr", font=dict(size=9)), thickness=8, len=0.5),
                margin=dict(l=40, r=10, t=10, b=40),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with cR:
            st.markdown(f"""
            <div class="hud">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title blue">Significance Layer</div>
                    <div class="hud-meta">FDR</div>
                </div>
                <div class="status-list">
                    <div class="status-item">
                        <span class="status-dot" style="background:{POS};box-shadow:0 0 6px {POS}"></span>
                        <span class="status-name">Significant</span>
                        <span class="status-val">{n_sig:,}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-dot" style="background:{NEUTRAL}"></span>
                        <span class="status-name">Not Sig.</span>
                        <span class="status-val">{n_cells-n_sig:,}</span>
                    </div>
                </div>
            </div>
            <div style="height:10px"></div>
            <div class="hud">
                <span class="hud-bl"></span><span class="hud-br"></span>
                <div class="hud-head">
                    <div class="hud-title violet">Slope Range</div>
                    <div class="hud-meta">MM/YR</div>
                </div>
                <div class="status-list">
                    <div class="status-item">
                        <span class="status-dot" style="background:{NEG}"></span>
                        <span class="status-name">Min</span>
                        <span class="status-val">{trends['sen_slope_mm_year'].min():.2f}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-dot" style="background:{CYAN}"></span>
                        <span class="status-name">Max</span>
                        <span class="status-val">{trends['sen_slope_mm_year'].max():.2f}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-dot" style="background:{MUTED}"></span>
                        <span class="status-name">Median</span>
                        <span class="status-val">{trends['sen_slope_mm_year'].median():.2f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title">Statistical Reading</div>
                <div class="hud-meta">CRITICAL</div>
            </div>
            <div class="mc-warn">
                <strong>No pixel passed Benjamini–Hochberg FDR correction (α = 0.05).</strong>
                Of {n_cells:,} grid cells, 0 are statistically distinguishable from noise after
                multiple-testing correction. The coloured Sen's slope map shows <em>estimated</em>
                local slopes only — individual pixels must not be described as confirmed increasing
                or decreasing rainfall trends.
            </div>
            <div style="height:8px"></div>
            <div class="mc-note">
                <strong>National vs pixel-level.</strong> The national aggregate Mann–Kendall result
                (p = 0.035, Sen's slope = +5.231 mm/yr) describes the spatially averaged series.
                The pixel tests ask whether individual grid cells show detectable trends after
                multiple-testing control. These are different questions — the national aggregate can
                be significant while zero pixels survive correction.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE: TERRAIN
# ============================================================================
elif page == "TERRAIN":
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Source</div>
            <div class="kpi-value" style="font-size:1rem;">SRTMGL1</div>
            <div class="kpi-sub">1 arc-second</div></div>
        <div class="kpi"><div class="kpi-label">Type</div>
            <div class="kpi-value" style="font-size:1rem;">STATIC</div>
            <div class="kpi-sub">context only</div></div>
        <div class="kpi"><div class="kpi-label">Resolution</div>
            <div class="kpi-value">1 km</div>
            <div class="kpi-sub">resampled</div></div>
        <div class="kpi"><div class="kpi-label">Coverage</div>
            <div class="kpi-value">PAK</div>
            <div class="kpi-sub">60–78 E · 22–38 N</div></div>
    </div>
    """, unsafe_allow_html=True)

    if not TERRAIN_FILE.exists():
        st.info("Optional terrain file not found. Add `FloodLens_Pakistan_Terrain_1km.tif` to data/.")
    elif rasterio is None:
        st.error("Install rasterio: `pip install rasterio`")
    else:
        with rasterio.open(TERRAIN_FILE) as src:
            elev = src.read(1).astype("float32")
            nodata = src.nodata
            bounds = src.bounds
        if nodata is not None:
            elev[elev == nodata] = np.nan
        step = max(1, int(max(elev.shape) / 150))
        small = elev[::step, ::step]

        st.markdown(f"""
        <div class="hud" style="padding-bottom:0;">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head">
                <div class="hud-title">Digital Elevation Model</div>
                <div class="hud-meta">MIN {np.nanmin(elev):.0f}m · MAX {np.nanmax(elev):.0f}m</div>
            </div>
        """, unsafe_allow_html=True)
        f = px.imshow(
            small, origin="upper", aspect="auto",
            color_continuous_scale=[[0, "#040810"], [0.2, "#1e3a5f"], [0.4, "#4a7c59"], [0.7, "#a8a06a"], [1, "#e8e8e8"]],
        )
        f.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title=dict(text="m", font=dict(size=9)), thickness=8, len=0.6),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# PAGE: METHODOLOGY
# ============================================================================
else:
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head"><div class="hud-title">Precipitation</div><div class="hud-meta">INPUT</div></div>
            <div style="font-size:0.8rem;color:{TEXT};margin-bottom:6px;font-weight:600;">NASA GPM IMERG V07</div>
            <div style="font-size:0.7rem;color:{MUTED};line-height:1.6;">
                Gridded monthly precipitation at 0.1° resolution, aggregated to a national series for
                Pakistan, 2001–2024.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head"><div class="hud-title blue">Elevation</div><div class="hud-meta">CONTEXT</div></div>
            <div style="font-size:0.8rem;color:{TEXT};margin-bottom:6px;font-weight:600;">USGS SRTMGL1</div>
            <div style="font-size:0.7rem;color:{MUTED};line-height:1.6;">
                1 arc-second elevation. Static geographic context — never used as a trend variable.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="hud">
            <span class="hud-bl"></span><span class="hud-br"></span>
            <div class="hud-head"><div class="hud-title violet">Statistics</div><div class="hud-meta">METHOD</div></div>
            <div style="font-size:0.8rem;color:{TEXT};margin-bottom:6px;font-weight:600;">MK · Sen · BH-FDR</div>
            <div style="font-size:0.7rem;color:{MUTED};line-height:1.6;">
                Mann–Kendall with Hamed–Rao correction, Sen's slope estimator, Benjamini–Hochberg FDR.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hud">
        <span class="hud-bl"></span><span class="hud-br"></span>
        <div class="hud-head">
            <div class="hud-title">Interpretation Limits</div>
            <div class="hud-meta">READ CAREFULLY</div>
        </div>
        <div class="mc-warn">
            <strong>What FloodLens does not claim.</strong>
            Rainfall trends are estimates, not forecasts. National aggregate and pixel-level results
            answer different questions. Zero pixels passed FDR correction at pixel scale. Satellite
            precipitation carries retrieval uncertainty. SRTM elevation is static context.
        </div>
    </div>
    """, unsafe_allow_html=True)
