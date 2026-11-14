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
ACCENT_CYAN = "#27D9F2"
ACCENT_ORANGE = "#FB7185"
ACCENT_NASA_BLUE = "#0B3D91"
TEXT_LIGHT = "#F1F5F9"
TEXT_MUTED = "#94A3B8"
GRID_FAINT = "rgba(255,255,255,0.08)"

# Palette requested for the future-outlook section
FUTURE_PANEL = "#141F30"
FUTURE_BG2 = "#101827"
FUTURE_CYAN = "#27D9F2"
FUTURE_BLUE = "#648CFF"
FUTURE_VIOLET = "#A78BFA"
FUTURE_TEXT = "#F1F5F9"
FUTURE_MUTED = "#94A3B8"
FUTURE_POS = "#34D399"
FUTURE_NEG = "#FB7185"

BG_MAIN = "#080D18"

# FIX: alert palette constants (used in CSS)
ALERT_INFO = "#27D9F2"
ALERT_WARN = "#FBBF24"
ALERT_OK = "#34D399"
ALERT_ERR = "#FB7185"


def inject_css():
    """Inject the whole mission-control theme in one place, forcing dark mode on widgets."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        /* -------- Global Variables -------- */
        :root {{
            --bg-0: {BG_MAIN};
            --bg-1: {FUTURE_BG2};
            --panel: {FUTURE_PANEL};
            --panel-border: rgba(39, 217, 242, 0.25);
            --cyan: {ACCENT_CYAN};
            --orange: {ACCENT_ORANGE};
            --nasa-blue: {ACCENT_NASA_BLUE};
            --text: {TEXT_LIGHT};
            --muted: {TEXT_MUTED};
        }}

        /* -------- App Background -------- */
        .stApp {{
            background:
                radial-gradient(1200px 600px at 15% -10%, rgba(11,61,145,0.35), transparent 60%),
                radial-gradient(900px 500px at 110% 20%, rgba(39,217,242,0.10), transparent 60%),
                linear-gradient(160deg, {BG_MAIN} 0%, {FUTURE_BG2} 100%);
            background-attachment: fixed;
            color: var(--text);
            font-family: 'Inter', sans-serif;
        }}

        /* -------- Hide default Streamlit chrome -------- */
        #MainMenu, header, footer {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1350px; }}

        /* -------- Typography -------- */
        h1, h2, h3, h4, h5, h6, .stSubheader {{
            font-family: 'Orbitron', 'Rajdhani', sans-serif !important;
            letter-spacing: 0.06em;
            color: var(--text) !important;
        }}
        p, span, label, div, .stMarkdown {{ color: var(--text); }}

        /* -------- Hero Header -------- */
        .fl-hero {{
            position: relative;
            padding: 26px 30px 22px 30px;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(11,61,145,0.35), rgba(39,217,242,0.06));
            border: 1px solid var(--panel-border);
            box-shadow: 0 0 30px rgba(39,217,242,0.10), inset 0 0 24px rgba(39,217,242,0.04);
            overflow: hidden;
            margin-bottom: 20px;
        }}
        .fl-title {{
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            font-size: 44px;
            letter-spacing: 0.32em;
            color: var(--text);
            text-shadow: 0 0 18px rgba(39,217,242,0.55), 0 0 4px rgba(39,217,242,0.9);
            margin: 0;
            line-height: 1.05;
        }}
        .fl-sub {{
            font-family: 'Rajdhani', sans-serif;
            font-weight: 500;
            font-size: 15px;
            letter-spacing: 0.14em;
            color: var(--muted);
            margin-top: 8px;
            text-transform: uppercase;
        }}
        .fl-divider {{
            margin: 16px 0 12px 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--cyan), transparent);
            box-shadow: 0 0 12px rgba(39,217,242,0.7);
            border: none;
        }}
        .fl-chips {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 4px; }}
        .fl-chip {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 12px;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(39,217,242,0.35);
            background: rgba(39,217,242,0.06);
            color: var(--cyan);
            box-shadow: 0 0 12px rgba(39,217,242,0.15);
        }}
        .fl-chip.orange {{
            border-color: rgba(251,113,133,0.35);
            background: rgba(251,113,133,0.06);
            color: #FDA4AF;
        }}

        /* -------- Future Panel & Disclaimers -------- */
        .fl-future-panel {{
            background: linear-gradient(160deg, {FUTURE_PANEL} 0%, {FUTURE_BG2} 100%);
            border: 1px solid rgba(39,217,242,0.25);
            border-radius: 16px;
            padding: 22px 24px;
            margin-bottom: 18px;
            box-shadow: 0 0 30px rgba(39,217,242,0.10);
        }}
        .fl-future-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 15px;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: {FUTURE_CYAN};
            margin-bottom: 16px;
        }}
        .fl-future-disclaimer {{
            background: rgba(251,113,133,0.08);
            border-left: 3px solid {FUTURE_NEG};
            padding: 12px 16px;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            line-height: 1.5;
            color: {FUTURE_TEXT};
            margin: 14px 0;
        }}

        /* -------- Widgets: Force Dark Theme Overrides -------- */
        /* Dropdowns, Text Inputs, Number Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stNumberInput"] > div > div {{
            background-color: {BG_MAIN} !important;
            border: 1px solid rgba(39,217,242,0.3) !important;
            border-radius: 8px !important;
            color: {TEXT_LIGHT} !important;
        }}

        /* FIX: open selectbox dropdown (popover/menu/listbox/options) */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="menu"],
        ul[role="listbox"] {{
            background-color: {FUTURE_PANEL} !important;
            color: {TEXT_LIGHT} !important;
            border: 1px solid {FUTURE_CYAN} !important;
            border-radius: 8px !important;
            box-shadow: 0 0 20px rgba(39,217,242,0.25) !important;
        }}
        ul[role="listbox"] li,
        li[role="option"] {{
            background-color: {FUTURE_PANEL} !important;
            color: {TEXT_LIGHT} !important;
        }}
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {{
            background-color: rgba(39,217,242,0.15) !important;
            color: {FUTURE_CYAN} !important;
        }}
        /* Preserve legacy selector used elsewhere in this file */
        ul[data-testid="stSelectboxVirtualDropdown"] {{
            background-color: {FUTURE_PANEL} !important;
            border: 1px solid {FUTURE_CYAN} !important;
        }}
        ul[data-testid="stSelectboxVirtualDropdown"] li {{
            color: {TEXT_LIGHT} !important;
        }}
        ul[data-testid="stSelectboxVirtualDropdown"] li:hover {{
            background-color: rgba(39,217,242,0.15) !important;
            color: {FUTURE_CYAN} !important;
        }}

        /* FIX: text inside closed select / inputs / svg icons */
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] svg,
        div[data-baseweb="input"] input,
        div[data-baseweb="input"] span,
        div[data-baseweb="input"] svg,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] svg {{
            color: {TEXT_LIGHT} !important;
            fill: {TEXT_LIGHT} !important;
        }}

        /* FIX: number input +/- stepper buttons */
        [data-testid="stNumberInputStepUp"],
        [data-testid="stNumberInputStepDown"] {{
            background-color: {BG_MAIN} !important;
            color: {FUTURE_CYAN} !important;
            border: 1px solid rgba(39,217,242,0.3) !important;
        }}
        [data-testid="stNumberInputStepUp"]:hover,
        [data-testid="stNumberInputStepDown"]:hover {{
            background-color: rgba(39,217,242,0.15) !important;
            color: #FFFFFF !important;
        }}

        /* FIX: file uploader dropzone + button */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {BG_MAIN} !important;
            border: 1px dashed rgba(39,217,242,0.45) !important;
            border-radius: 10px !important;
            color: {TEXT_LIGHT} !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {TEXT_LIGHT} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: rgba(39,217,242,0.10) !important;
            color: {FUTURE_CYAN} !important;
            border: 1px solid rgba(39,217,242,0.4) !important;
            border-radius: 8px !important;
        }}

        /* FIX: cyan focus glow on all inputs */
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within,
        div[data-testid="stNumberInput"] > div > div:focus-within,
        div[data-testid="stTextInput"] input:focus,
        textarea:focus {{
            border-color: {FUTURE_CYAN} !important;
            box-shadow: 0 0 0 2px rgba(39,217,242,0.35) !important;
        }}

        /* Expander */
        .stExpander {{
            border: 1px solid rgba(39,217,242,0.25) !important;
            border-radius: 12px !important;
            background: {FUTURE_PANEL} !important;
        }}
        .stExpander summary {{
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.1em;
            color: {FUTURE_CYAN} !important;
            background-color: transparent !important;
        }}
        .stExpander details {{
            background-color: {BG_MAIN} !important;
        }}

        /* Dataframe: keep the wrapper border/radius/padding; no attempt to recolour
           the canvas grid via CSS (see .fl-table for CSS-rendered tables) */
        div[data-testid="stDataFrame"] {{
            background-color: {FUTURE_PANEL} !important;
            border: 1px solid rgba(39,217,242,0.2);
            border-radius: 12px;
            padding: 8px;
        }}

        /* FIX: CSS-rendered dark table (used by render_dark_table) */
        .fl-table-wrap {{
            overflow-x: auto;
            border: 1px solid rgba(39,217,242,0.25);
            border-radius: 12px;
            background-color: {FUTURE_PANEL};
            padding: 4px;
            margin: 4px 0 12px 0;
        }}
        table.fl-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Inter', sans-serif;
            font-size: 13px;
            color: {TEXT_LIGHT};
            background-color: {FUTURE_PANEL};
        }}
        table.fl-table thead th {{
            background-color: {BG_MAIN};
            color: {FUTURE_CYAN};
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 12px;
            padding: 10px 12px;
            border-bottom: 1px solid {FUTURE_CYAN};
            text-align: left;
            white-space: nowrap;
        }}
        table.fl-table tbody td {{
            padding: 8px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            color: {TEXT_LIGHT};
            vertical-align: top;
        }}
        table.fl-table tbody tr:hover td {{
            background-color: rgba(39,217,242,0.06);
        }}
        table.fl-table td.num,
        table.fl-table th.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        table.fl-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        /* FIX: alerts (info / warning / success / error) */
        [data-testid="stAlert"],
        [data-testid="stAlertContainer"] {{
            background-color: rgba(20,31,48,0.75) !important;
            color: {TEXT_LIGHT} !important;
            border: 1px solid rgba(39,217,242,0.25) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
        }}
        [data-testid="stAlert"] *,
        [data-testid="stAlertContainer"] *,
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        [data-testid="stAlert"] li,
        [data-testid="stAlert"] strong,
        [data-testid="stAlert"] code,
        [data-testid="stAlertContainer"] p,
        [data-testid="stAlertContainer"] span,
        [data-testid="stAlertContainer"] li,
        [data-testid="stAlertContainer"] strong,
        [data-testid="stAlertContainer"] code {{
            color: {TEXT_LIGHT} !important;
        }}
        /* Alert variants (Streamlit tags containers with these kinds) */
        [data-testid="stAlert"][data-baseweb="notification"]:has(svg[aria-label="info"]),
        [data-testid="stAlert"]:has([aria-label="info"]) {{
            border-left: 4px solid {ALERT_INFO} !important;
            background-color: rgba(39,217,242,0.10) !important;
        }}
        [data-testid="stAlert"]:has([aria-label="warning"]) {{
            border-left: 4px solid {ALERT_WARN} !important;
            background-color: rgba(251,191,36,0.10) !important;
        }}
        [data-testid="stAlert"]:has([aria-label="success"]) {{
            border-left: 4px solid {ALERT_OK} !important;
            background-color: rgba(52,211,153,0.10) !important;
        }}
        [data-testid="stAlert"]:has([aria-label="error"]) {{
            border-left: 4px solid {ALERT_ERR} !important;
            background-color: rgba(251,113,133,0.10) !important;
        }}
        /* Fallback for Streamlit's stAlert-<kind> class variants */
        .stAlert-info, .stAlert-container.stAlert-info {{
            border-left: 4px solid {ALERT_INFO} !important;
            background-color: rgba(39,217,242,0.10) !important;
        }}
        .stAlert-warning, .stAlert-container.stAlert-warning {{
            border-left: 4px solid {ALERT_WARN} !important;
            background-color: rgba(251,191,36,0.10) !important;
        }}
        .stAlert-success, .stAlert-container.stAlert-success {{
            border-left: 4px solid {ALERT_OK} !important;
            background-color: rgba(52,211,153,0.10) !important;
        }}
        .stAlert-error, .stAlert-container.stAlert-error {{
            border-left: 4px solid {ALERT_ERR} !important;
            background-color: rgba(251,113,133,0.10) !important;
        }}

        /* FIX: st.code blocks readable on dark */
        [data-testid="stCode"],
        [data-testid="stCodeBlock"],
        div.highlight,
        pre {{
            background-color: {BG_MAIN} !important;
            color: {TEXT_LIGHT} !important;
            border: 1px solid rgba(39,217,242,0.20) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stCode"] code,
        [data-testid="stCodeBlock"] code,
        pre code {{
            color: {TEXT_LIGHT} !important;
            background-color: transparent !important;
        }}
        /* FIX: st.caption readable on dark */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .stCaption {{
            color: {TEXT_MUTED} !important;
        }}

        /* Metrics (KPI cards) */
        div[data-testid="stMetric"] {{
            background: {FUTURE_PANEL};
            border: 1px solid rgba(39,217,242,0.3);
            border-left: 4px solid {FUTURE_CYAN};
            border-radius: 12px;
            padding: 14px 16px 12px 16px;
            box-shadow: 0 0 18px rgba(39,217,242,0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 0 28px rgba(39,217,242,0.22);
            border-left-color: {FUTURE_BLUE};
        }}
        div[data-testid="stMetricLabel"] > div {{
            font-family: 'Rajdhani', sans-serif !important;
            font-size: 11px !important;
            letter-spacing: 0.18em !important;
            text-transform: uppercase !important;
            color: {FUTURE_MUTED} !important;
        }}
        div[data-testid="stMetricValue"] > div {{
            font-family: 'Orbitron', sans-serif !important;
            font-size: 26px !important;
            font-weight: 700 !important;
            color: {FUTURE_CYAN} !important;
            text-shadow: 0 0 12px rgba(39,217,242,0.45);
        }}

        /* -------- Buttons -------- */
        .stButton>button, .stDownloadButton>button {{
            font-family: 'Rajdhani', sans-serif;
            letter-spacing: 0.1em;
            background: rgba(39,217,242,0.08);
            border: 1px solid rgba(39,217,242,0.4);
            color: {FUTURE_CYAN};
            border-radius: 10px;
            transition: all 0.2s;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            background: rgba(39,217,242,0.18);
            box-shadow: 0 0 18px rgba(39,217,242,0.3);
            color: #fff;
            border-color: {FUTURE_CYAN};
        }}

        /* -------- Sidebar -------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #070E1F 0%, {BG_MAIN} 100%);
            border-right: 1px solid rgba(39,217,242,0.15);
        }}
        .fl-sidebar-logo {{
            font-family: 'Orbitron', sans-serif;
            font-size: 22px;
            font-weight: 900;
            letter-spacing: 0.28em;
            color: var(--text);
            text-shadow: 0 0 14px rgba(39,217,242,0.6);
            margin-bottom: 2px;
        }}
        .fl-sidebar-tag {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 11px;
            letter-spacing: 0.22em;
            color: var(--cyan);
            text-transform: uppercase;
            margin-bottom: 18px;
        }}
        /* Sidebar Menu Radio overrides */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
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
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
            background: rgba(39,217,242,0.06);
            border-color: rgba(39,217,242,0.35);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {{
            background: linear-gradient(90deg, rgba(39,217,242,0.16), rgba(39,217,242,0.02));
            border-color: var(--cyan);
            box-shadow: 0 0 20px rgba(39,217,242,0.28);
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] * {{
            color: var(--cyan) !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
            display: none; /* Hide default dot */
        }}

        /* -------- Misc Components -------- */
        .fl-card {{
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 20px 22px;
            margin-bottom: 18px;
        }}
        .fl-card-title {{
            font-family: 'Orbitron', sans-serif;
            font-size: 13px;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--cyan);
            margin-bottom: 14px;
        }}
        .fl-hud-caption {{
            font-family: 'Rajdhani', sans-serif;
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted);
            margin: -8px 0 18px 2px;
            padding-left: 10px;
            border-left: 2px solid var(--cyan);
        }}
        .fl-section {{
            font-family: 'Orbitron', sans-serif;
            font-size: 14px;
            letter-spacing: 0.24em;
            text-transform: uppercase;
            color: var(--cyan);
            margin: 24px 0 14px 2px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(39,217,242,0.2);
        }}
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
            bgcolor=FUTURE_PANEL,
            bordercolor=FUTURE_CYAN,
            font=dict(family="Inter, sans-serif", color=TEXT_LIGHT, size=12),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(39,217,242,0.2)",
            font=dict(color=TEXT_LIGHT),
        ),
    )
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


# FIX: render_dark_table — themed replacement for st.dataframe
def render_dark_table(df, decimals=2, num_cols=None, index=False):
    """Render a pandas DataFrame as a themed dark HTML table with horizontal scroll.

    Numeric columns are right-aligned and formatted to `decimals` decimal places,
    unless explicitly listed in `num_cols` (also supports columns to *exclude* by
    omitting from that list). Non-numeric columns are rendered as-is.
    """
    if df is None or len(df) == 0:
        st.info("No rows to display.")
        return

    work = df.copy()

    # Determine numeric columns to format
    if num_cols is None:
        numeric_cols = list(work.select_dtypes(include=["number"]).columns)
    else:
        numeric_cols = [c for c in num_cols if c in work.columns]

    styler = work.style

    def _fmt_num(v):
        if pd.isna(v):
            return ""
        try:
            return f"{float(v):,.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    for col in numeric_cols:
        styler = styler.format({col: _fmt_num})

    html = styler.hide(axis="index").to_html(
        classes="fl-table",
        index=index,
        border=0,
        escape=False,
    )
    st.markdown(f'<div class="fl-table-wrap">{html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data paths & loaders
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
    return pd.DataFrame({
        "year": [int(forecast_year)] * len(regions),
        "region": list(regions),
        "pre_rainfall_mm": [np.nan] * len(regions),
        "pre_soil_moisture_mm": [np.nan] * len(regions),
    })


def run_regional_forecast(history, performance, future_inputs, alpha=10.0):
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


def add_boundary_lines(fig, geojson, color="#27D9F2", width=1.4):
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
        st.error("Monthly CSV needs columns named 'date' and 'monthly_rainfall_mm'.")
        return None
    df = df[["date", "monthly_rainfall_mm"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["monthly_rainfall_mm"] = pd.to_numeric(df["monthly_rainfall_mm"], errors="coerce")
    return df.dropna().sort_values("date")


def load_trends():
    if not TRENDS_FILE.exists():
        return None
    df = read_csv(TRENDS_FILE)
    required = {"latitude", "longitude", "sen_slope_mm_year"}
    if not required.issubset(df.columns):
        st.error("Trend CSV must include latitude, longitude and sen_slope_mm_year columns.")
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
        df["significant_fdr"] = df["significant"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["significant_fdr"] = False
    return df.dropna(subset=["latitude", "longitude", "sen_slope_mm_year"])


def make_trend_map(trends, boundary, mode):
    plot_df = trends.copy()
    if mode == "Statistical significance":
        plot_df["status"] = np.where(
            plot_df["significant_fdr"], "Significant after FDR", "Not significant after FDR"
        )
        fig = px.scatter(
            plot_df, x="longitude", y="latitude", color="status",
            color_discrete_map={
                "Significant after FDR": FUTURE_POS,
                "Not significant after FDR": FUTURE_NEG,
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
            plot_df, x="longitude", y="latitude", color="sen_slope_mm_year",
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
# Regional GPM outlook helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_regional_gpm_history():
    if not FORECAST_DATA_FILE.exists():
        return None
    df = read_csv(FORECAST_DATA_FILE)
    required = {"year", "region", "monsoon_rainfall_mm"}
    if not required.issubset(df.columns):
        return None
    out = df[["year", "region", "monsoon_rainfall_mm"]].copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["monsoon_rainfall_mm"] = pd.to_numeric(out["monsoon_rainfall_mm"], errors="coerce")
    out = out.dropna(subset=["year", "region", "monsoon_rainfall_mm"])
    out["year"] = out["year"].astype(int)
    return out.sort_values(["region", "year"]).reset_index(drop=True)


def fit_linear_trend(series_years, series_values):
    years = np.asarray(series_years, dtype=float)
    values = np.asarray(series_values, dtype=float)
    if len(years) < 2 or np.allclose(years, years[0]):
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(years, values, 1)
    predicted = slope * years + intercept
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(r2)


def compute_projection(regional_hist, region, target_year, method):
    sub = regional_hist[regional_hist["region"] == region].sort_values("year")
    if sub.empty:
        return None
    years = sub["year"].to_numpy()
    values = sub["monsoon_rainfall_mm"].to_numpy()
    n = len(years)
    slope, intercept, r2 = fit_linear_trend(years, values)

    if method == "Linear trend projection":
        estimate = slope * target_year + intercept if not np.isnan(slope) else np.nan
    elif method == "Recent five-year average":
        last5 = values[-5:] if n >= 1 else values
        estimate = float(np.mean(last5))
    else:
        estimate = float(np.mean(values))

    recent5 = float(np.mean(values[-5:])) if n >= 1 else np.nan
    return {
        "region": region,
        "method": method,
        "target_year": int(target_year),
        "estimated_mm": float(max(estimate, 0.0)) if not np.isnan(estimate) else np.nan,
        "trend_mm_per_year": slope,
        "trend_r2": r2,
        "historical_mean_mm": float(np.mean(values)),
        "recent5_mean_mm": recent5,
        "n_years": int(n),
        "first_year": int(years.min()),
        "last_year": int(years.max()),
    }


def build_future_outlook_chart(regional_hist, region, target_year, method):
    sub = regional_hist[regional_hist["region"] == region].sort_values("year")
    if sub.empty:
        return None
    years = sub["year"].to_numpy().astype(float)
    values = sub["monsoon_rainfall_mm"].to_numpy().astype(float)
    slope, intercept, _ = fit_linear_trend(years, values)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years, y=values, mode="lines+markers", name="Observed GPM Jun–Sep total",
        line=dict(color=FUTURE_CYAN, width=2.4),
        marker=dict(size=8, color=FUTURE_CYAN, line=dict(color="#FFFFFF", width=1)),
    ))

    if not np.isnan(slope):
        trend_years = np.array([years.min(), years.max()])
        trend_vals = slope * trend_years + intercept
        fig.add_trace(go.Scatter(
            x=trend_years, y=trend_vals, mode="lines",
            name=f"Linear trend ({slope:+.2f} mm/year)",
            line=dict(color=FUTURE_VIOLET, width=2.2, dash="dash"),
        ))

        if target_year > years.max():
            ext_years = np.array([years.max(), float(target_year)])
            ext_vals = slope * ext_years + intercept
            fig.add_trace(go.Scatter(
                x=ext_years, y=ext_vals, mode="lines+markers",
                name=f"Trend extension → {int(target_year)}",
                line=dict(color=FUTURE_BLUE, width=2.4, dash="dot"),
                marker=dict(size=10, symbol="diamond", color=FUTURE_BLUE),
            ))

    recent5 = float(np.mean(values[-5:]))
    fig.add_trace(go.Scatter(
        x=[years.min(), max(years.max(), float(target_year))], y=[recent5, recent5],
        mode="lines", name=f"Recent 5-yr mean ({recent5:.0f} mm)",
        line=dict(color=FUTURE_POS, width=1.5, dash="longdash"),
    ))

    hist_mean = float(np.mean(values))
    fig.add_trace(go.Scatter(
        x=[years.min(), max(years.max(), float(target_year))], y=[hist_mean, hist_mean],
        mode="lines", name=f"Historical mean ({hist_mean:.0f} mm)",
        line=dict(color=FUTURE_MUTED, width=1.2, dash="dot"),
    ))

    fig.add_vline(
        x=float(years.max()),
        line=dict(color=FUTURE_NEG, width=1.5, dash="dash"),
        annotation_text="Last historical obs.",
        annotation_position="top",
        annotation_font=dict(color=FUTURE_NEG, size=11),
    )

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="June–September rainfall (mm)",
        legend=dict(orientation="h", y=-0.2, x=0),
        height=470,
    )
    # FIX: integer year ticks on future outlook chart
    fig.update_xaxes(tickmode="linear", dtick=1, tickformat="d")
    return style_fig(fig, height=470)


def render_future_outlook(regional_hist):
    st.markdown(
        f'<div class="fl-future-panel">'
        f'<div class="fl-future-title">🛰️ Future Monsoon Outlook – NASA GPM Historical Projections</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="fl-future-disclaimer">'
        "These estimates are exploratory statistical projections based on historical "
        "NASA GPM rainfall observations. They assume that historical rainfall patterns "
        "remain informative about future conditions. Natural climate variability and "
        "changing environmental conditions may produce different outcomes. These "
        "estimates are not operational weather forecasts or flood warnings."
        "</div>",
        unsafe_allow_html=True,
    )

    regions = sorted(regional_hist["region"].dropna().unique().tolist())
    if not regions:
        st.warning("No regional historical GPM rainfall records available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    last_year = int(regional_hist["year"].max())
    default_target = max(2027, last_year + 1)
    target_min = max(last_year + 1, 2027)
    target_max = max(target_min, 2035)

    # Use columns for the selectors to make them align horizontally
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        selected_region = st.selectbox(
            "Study region", regions,
            index=regions.index("Punjab and upper Indus") if "Punjab and upper Indus" in regions else 0,
            key="future_outlook_region",
        )
    with c2:
        target_year = st.number_input(
            "Future year", min_value=int(target_min), max_value=int(target_max),
            value=int(min(default_target, target_max)), step=1, key="future_outlook_year",
        )
    with c3:
        method = st.selectbox(
            "Projection method",
            ["Linear trend projection", "Recent five-year average", "Long-term historical average"],
            key="future_outlook_method",
        )

    method_notes = {
        "Linear trend projection": "Assumes the historical linear trend continues unchanged into the future.",
        "Recent five-year average": "Reference estimate based on the most recent 5 observed years. Not a forecast.",
        "Long-term historical average": "Climatological reference: mean of the complete available historical record.",
    }
    st.caption(f"**Method note:** {method_notes[method]}")

    proj = compute_projection(regional_hist, selected_region, int(target_year), method)
    if proj is None:
        st.error("Could not compute projection for the selected region.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Estimated Jun–Sep rainfall ({int(target_year)})", f'{proj["estimated_mm"]:,.0f} mm')
    k2.metric("Historical trend", f'{proj["trend_mm_per_year"]:+.2f} mm/yr' if not np.isnan(proj["trend_mm_per_year"]) else "n/a")
    k3.metric("Recent 5-yr average", f'{proj["recent5_mean_mm"]:,.0f} mm')
    k4.metric("Historical average", f'{proj["historical_mean_mm"]:,.0f} mm')

    if not np.isnan(proj["trend_r2"]):
        st.caption(f"Historical record: **{proj['first_year']}–{proj['last_year']}** (**{proj['n_years']} years** used) · trend R² = {proj['trend_r2']:.3f}")
    else:
        st.caption(f"Historical record: **{proj['first_year']}–{proj['last_year']}** (**{proj['n_years']} years** used)")

    # Chart
    chart = build_future_outlook_chart(regional_hist, selected_region, int(target_year), method)
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True, key="future_outlook_chart")
    hud_caption("OBSERVED GPM JUN–SEP TOTALS · TREND · RECENT-5 MEAN · HISTORICAL MEAN · TARGET-YEAR EXTENSION")

    # Comparison table
    st.markdown('<div class="fl-card-title" style="margin-top:14px;">All-region projection comparison</div>', unsafe_allow_html=True)
    rows = []
    for r in regions:
        p = compute_projection(regional_hist, r, int(target_year), method)
        if p is not None:
            rows.append({
                "Region": r,
                f"Estimated {int(target_year)} (mm)": round(p["estimated_mm"], 1) if not np.isnan(p["estimated_mm"]) else None,
                "Trend (mm/yr)": round(p["trend_mm_per_year"], 3) if not np.isnan(p["trend_mm_per_year"]) else None,
                "Recent 5-yr mean (mm)": round(p["recent5_mean_mm"], 1),
                "Historical mean (mm)": round(p["historical_mean_mm"], 1),
                "Years used": p["n_years"],
            })
    comp_df = pd.DataFrame(rows)

    # FIX: replace st.dataframe with themed table
    render_dark_table(comp_df, decimals=2)

    if not comp_df.empty:
        st.download_button(
            "⬇️ Download projection results (CSV)",
            data=comp_df.to_csv(index=False).encode("utf-8"),
            file_name=f"FloodLens_Monsoon_Projections_{int(target_year)}_{method.replace(' ', '_')}.csv",
            mime="text/csv",
            key="future_outlook_download",
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Boot the theme
# ---------------------------------------------------------------------------
inject_css()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
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
# Sidebar
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

        st.markdown('<div class="fl-card"><div class="fl-card-title">Telemetry · Dataset</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Monthly observations", f"{len(monthly):,}")
        c2.metric("Years covered", f"{monthly['date'].dt.year.min()}–{monthly['date'].dt.year.max()}")
        c3.metric("Mean annual rainfall", f"{annual['annual_rainfall_mm'].mean():,.1f} mm")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fl-card"><div class="fl-card-title">Annual Signal</div>', unsafe_allow_html=True)
        fig = px.line(
            annual, x="year", y="annual_rainfall_mm",
            markers=True, title="Pakistan annual rainfall total",
        )
        fig.update_traces(
            line=dict(color=ACCENT_CYAN, width=2.4, shape="spline"),
            marker=dict(size=8, color=ACCENT_CYAN, line=dict(width=1.5, color="white")),
            fill="tozeroy",
            fillcolor="rgba(39,217,242,0.12)",
        )
        fig.update_layout(xaxis_title="Year", yaxis_title="Annual rainfall (mm)")
        fig = style_fig(fig, height=460)
        # FIX: integer year ticks on annual chart
        fig.update_xaxes(tickmode="linear", dtick=1, tickformat="d")
        st.plotly_chart(fig, use_container_width=True)
        hud_caption("ANNUAL NATIONAL RAINFALL TOTALS · GPM IMERG MONTHLY V07")
        st.markdown("</div>", unsafe_allow_html=True)

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

        st.markdown('<div class="fl-card"><div class="fl-card-title">Control Panel</div>', unsafe_allow_html=True)
        selected_years = st.slider(
            "Select year range", min_year, max_year, (min_year, max_year)
        )
        st.markdown("</div>", unsafe_allow_html=True)

        subset = monthly[
            monthly["year"].between(selected_years[0], selected_years[1])
        ]

        st.markdown('<div class="fl-card"><div class="fl-card-title">Monthly Signal</div>', unsafe_allow_html=True)
        fig = px.line(
            subset, x="date", y="monthly_rainfall_mm",
            title="Monthly rainfall through time",
        )
        fig.update_traces(
            line=dict(color=ACCENT_CYAN, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(39,217,242,0.10)",
        )
        fig.update_layout(xaxis_title="Month", yaxis_title="Rainfall (mm)")
        fig = style_fig(fig, height=430)
        st.plotly_chart(fig, use_container_width=True)
        hud_caption("MONTHLY RAINFALL THROUGH TIME · GPM IMERG")
        st.markdown("</div>", unsafe_allow_html=True)

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
                [0.8, ACCENT_CYAN],
                [1.0, "#FFFFFF"],
            ],
            labels=dict(x="Month", y="Year", color="Rainfall (mm)"),
            title="Monthly rainfall heatmap",
        )
        fig_heat = style_fig(fig_heat, height=480)
        st.plotly_chart(fig_heat, use_container_width=True)
        hud_caption("CALENDAR-MONTH MEAN RAINFALL · DEEP-NAVY → CYAN → WHITE SCALE")
        st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown('<div class="fl-card"><div class="fl-card-title">Control Panel</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Map layer",
            ["Sen's slope (mm/year)", "Statistical significance"],
            horizontal=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="fl-card"><div class="fl-card-title">Pixel Map</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_trend_map(trends, boundary, mode),
            use_container_width=True,
        )
        hud_caption("PIXEL-LEVEL SEN'S SLOPE · PAKISTAN BOUNDARY OVERLAY")
        st.markdown("</div>", unsafe_allow_html=True)

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

        st.markdown('<div class="fl-card"><div class="fl-card-title">Telemetry · Elevation</div>', unsafe_allow_html=True)
        st.metric(
            "Raster elevation range",
            f"{np.nanmin(elevation):.0f} to {np.nanmax(elevation):.0f} m",
        )
        st.markdown("</div>", unsafe_allow_html=True)

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
# PAGE: Regional monsoon outlook  (UPGRADED & FIXED)
# ---------------------------------------------------------------------------
elif page == "Regional monsoon outlook":
    st.markdown(
        '<div class="fl-section">Regional Monsoon Outlook</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 🌧️ Regional monsoon outlook")

    forecast_data, validation, performance, missing_forecast_files = load_forecast_files()

    # 1. PRIMARY FEATURE: Future Monsoon Outlook
    regional_hist = load_regional_gpm_history()
    if regional_hist is not None and not regional_hist.empty:
        render_future_outlook(regional_hist)
    else:
        st.warning(
            "The regional historical GPM rainfall file "
            "(FloodLens_Regional_Forecast_Dataset.csv) was not found or is missing "
            "required columns (year, region, monsoon_rainfall_mm). "
            "Add it to the data/ folder to enable the future outlook."
        )

    # 2. PRESERVED: Historical Ridge Experiment
    st.markdown(
        '<div class="fl-section">Historical Ridge Experiment & Backtesting (2001–2021)</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "This section preserves the original Ridge-regression experiment and "
        "its 2017–2021 backtest. It is optional and advanced; the primary future "
        "outlook above does not depend on it."
    )

    if missing_forecast_files:
        st.warning(
            "Add these three files to the app's data/ folder to enable this section: "
            + ", ".join(missing_forecast_files)
        )
        st.code(
            "data/FloodLens_Regional_Forecast_Dataset.csv\n"
            "data/FloodLens_Forecast_Validation.csv\n"
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
                key="ridge_region_select",
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

            # Optional Ridge-based upload tool
            with st.expander("🔬 Advanced · Upload March–May predictors for Ridge experiment"):
                st.write(
                    "Upload March–May predictor values for a future monsoon year. "
                    "The app will fit the same historical Ridge experiment and "
                    "apply the region-specific method selected by the backtest."
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
                                display_forecast = forecast_result[[
                                    "year", "region", "pre_rainfall_mm", "pre_soil_moisture_mm",
                                    "ridge_prediction_mm", "baseline_prediction_mm",
                                    "recommended_method", "selected_prediction_mm"
                                ]].sort_values("region")
                                # FIX: replace st.dataframe with themed table
                                render_dark_table(
                                    display_forecast.round(2),
                                    decimals=2,
                                )
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

            # Historical validation summary (unchanged)
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
            # FIX: integer year ticks on historical monsoon chart
            fig_history.update_xaxes(tickmode="linear", dtick=1, tickformat="d")
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
                # FIX: integer year ticks on validation chart
                fig_validation.update_xaxes(tickmode="linear", dtick=1, tickformat="d")
                st.plotly_chart(
                    style_fig(fig_validation, height=420),
                    use_container_width=True,
                )
                # FIX: replace st.dataframe with themed table
                render_dark_table(
                    region_validation[
                        ["year", "actual", "ridge_prediction", "baseline_prediction"]
                    ].round(2),
                    decimals=2,
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
            # FIX: wrap " and " for shorter x labels to prevent overlap
            wrapped_labels = [str(r).replace(" and ", "<br>and ") for r in comparison["region"]]
            fig_compare.update_layout(
                barmode="group",
                xaxis_title="Study region",
                yaxis_title="Mean absolute error (mm)",
                margin=dict(l=20, r=20, t=55, b=90),
            )
            fig_compare.update_xaxes(tickangle=-20, ticktext=wrapped_labels, tickvals=list(comparison["region"]))
            st.plotly_chart(
                style_fig(fig_compare, height=420),
                use_container_width=True,
            )
            # FIX: replace st.dataframe with themed table
            render_dark_table(
                comparison[
                    ["region", "Ridge_MAE", "Baseline_MAE",
                     "Improvement_percent", "Recommended_method"]
                ].round(2),
                decimals=2,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.warning(
                "Interpretation: validation covers only five held-out years "
                "(2017–2021). Regional performance is preliminary. The model "
                "must not be interpreted as an operational flood-warning system."
            )

            with st.expander("Inspect the model input dataset"):
                # FIX: replace st.dataframe with themed table
                render_dark_table(
                    forecast_data[forecast_data["region"] == selected_region],
                    decimals=2,
                )

            st.download_button(
                "Download regional forecast dataset",
                data=forecast_data.to_csv(index=False).encode("utf-8"),
                file_name="FloodLens_Regional_Forecast_Dataset.csv",
                mime="text/csv",
                key="download_region_dataset",
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
        - The **Future Monsoon Outlook** is an exploratory statistical projection
          based on historical NASA GPM regional monsoon totals. It is not an
          operational forecast, nor is it a flood prediction.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="fl-card"><div class="fl-card-title">Data Files Expected in data/</div>', unsafe_allow_html=True)
    st.code(
        "Pakistan_GPM_Rainfall_2001_2024.csv\n"
        "FloodLens_Pakistan_Pixel_Trends_2001_2024.csv\n"
        "pakistan_boundary.geojson\n"
        "FloodLens_Pakistan_Terrain_1km.tif  # optional\n"
        "FloodLens_Regional_Forecast_Dataset.csv\n"
        "FloodLens_Forecast_Validation.csv\n"
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
