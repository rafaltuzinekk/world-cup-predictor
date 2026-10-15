"""Mistrzostwa Świata 2026 - dashboard analityczny fazy pucharowej.

Aplikacja Streamlit zbudowana na module ``src``: model hybrydowy (Elo + forma + styl gry),
deterministyczna drabinka z trybem Sandbox, analiza head-to-head, symulacja Monte Carlo
oraz światowy ranking Elo.

Uruchomienie:  streamlit run app.py
"""
from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.bracket_simulator import (
    MONTE_CARLO_CSV,
    ROUND_LABELS,
    prune_stale_overrides,
    run_monte_carlo,
    simulate_bracket,
)
from src.database import DB_PATH, get_connection
from src.elo_engine import CURRENT_ELO_CSV
from src.flags import flag_emoji, flag_url
from src.ml_predictor import HOST_BONUS, STAGES, HybridMatchPredictor


def _streamlit_has_width_stretch() -> bool:
    """``width="stretch"`` (zamiast ``use_container_width=True``) działa od Streamlit 1.50."""
    try:
        from packaging.version import Version

        return Version(st.__version__) >= Version("1.50")
    except Exception:  # nieparsowalna wersja -> wariant działający wszędzie
        return False


# Kompatybilność ze Streamlit 1.38+ (Community Cloud) i najnowszym (use_container_width jest tam przestarzałe)
STRETCH = {"width": "stretch"} if _streamlit_has_width_stretch() else {"use_container_width": True}

st.set_page_config(
    page_title="Mistrzostwa Świata 2026 | Analityka fazy pucharowej",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Stałe
# --------------------------------------------------------------------------- #
MODE_LABELS: Dict[str, str] = {
    "actual_plus_ai": "Rzeczywisty przebieg + Predykcja AI",
    "pre_tournament": "Prognoza przed fazą pucharową (100% AI)",
    "sandbox": "Tryb Sandbox (Własny scenariusz)",
}
MODE_HELP = (
    "Rzeczywisty przebieg: rozegrane mecze mają wynik faktyczny, resztę prognozuje model. "
    "Prognoza przed fazą pucharową: cała drabinka od 1/16 finału jest prognozą. "
    "Sandbox: klikasz drużynę bezpośrednio w drabince, aby ją awansować; reszta drabinki przelicza się na żywo."
)
SLOT_HEIGHT = 88  # px na jeden mecz 1/16 finału w drabince
MC_SIMULATIONS = 10_000

COL_BLUE, COL_ORANGE, COL_GOLD, COL_PURPLE = "#4C9AFF", "#F0A35C", "#E3B341", "#A78BFA"
COL_GREEN, COL_RED, COL_TEXT, COL_MUTED, COL_GRID = "#2FBF8F", "#EF6461", "#E6EAF2", "#8892A6", "#232D44"

# Polskie nazwy do wyszukiwarki rankingu (dane są w konwencji angielskiej)
POLISH_ALIASES: Dict[str, str] = {
    "polska": "Poland", "niemcy": "Germany", "francja": "France", "hiszpania": "Spain",
    "włochy": "Italy", "wlochy": "Italy", "anglia": "England", "brazylia": "Brazil",
    "argentyna": "Argentina", "holandia": "Netherlands", "niderlandy": "Netherlands",
    "portugalia": "Portugal", "belgia": "Belgium", "chorwacja": "Croatia",
    "szwajcaria": "Switzerland", "szwecja": "Sweden", "czechy": "Czech Republic",
    "ukraina": "Ukraine", "austria": "Austria", "dania": "Denmark", "norwegia": "Norway",
    "szkocja": "Scotland", "walia": "Wales", "irlandia": "Republic of Ireland",
    "turcja": "Turkey", "grecja": "Greece", "węgry": "Hungary", "wegry": "Hungary",
    "japonia": "Japan", "korea": "South Korea", "meksyk": "Mexico", "kanada": "Canada",
    "stany": "United States", "usa": "United States", "maroko": "Morocco",
    "egipt": "Egypt", "kolumbia": "Colombia", "urugwaj": "Uruguay",
}

# --------------------------------------------------------------------------- #
# Style (CSS)
# --------------------------------------------------------------------------- #
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
/* Czcionka Inter dla całego UI - z wyłączeniem ikon Material (przycisk zwijania sidebara, strzałki
   expanderów itd.), które są zwykłym tekstem renderowanym czcionką-ligaturą "Material Symbols". */
.stApp,
.stApp :where(p, label, button, input, textarea, li, td, th, div, span, h1, h2, h3, h4):not(
    [data-testid="stIconMaterial"], [data-testid^="stIcon"], [data-testid="stExpanderIcon"],
    [class*="material" i], [translate="no"]
) {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
/* Wymuszenie czcionki ikon (także, gdy nadrzędna reguła lub inna wersja Streamlita ustawi inną rodzinę) */
.stApp [data-testid="stIconMaterial"],
.stApp [data-testid="stSidebarCollapseButton"] [translate="no"],
.stApp [data-testid="stExpandSidebarButton"] [translate="no"],
.stApp [data-testid="stSidebarCollapsedControl"] [translate="no"],
.stApp [data-testid="stExpanderIcon"],
.stApp .material-icons, .stApp .material-symbols-rounded, .stApp [class*="material-symbols" i] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    font-feature-settings: "liga";
    letter-spacing: normal;
    text-transform: none;
}
.stApp { background: #0B0F17; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1600px; }
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

[data-testid="stSidebar"] { background: #0E1422; border-right: 1px solid #232D44; }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.side-title { font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: #8892A6; font-weight: 600; margin: 2px 0 10px; }
.side-brand { font-size: 17px; font-weight: 700; color: #F2F5FA; line-height: 1.25; margin-bottom: 2px; }
.side-sub { font-size: 12.5px; color: #8892A6; margin-bottom: 14px; }
.legend-row { display: flex; align-items: center; gap: 9px; font-size: 12.5px; color: #C3CBDA; margin: 6px 0; }
.legend-dot { width: 10px; height: 10px; border-radius: 3px; flex: none; }
.side-note { font-size: 12px; color: #8892A6; line-height: 1.55; }

.hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap;
        padding: 4px 2px 18px; border-bottom: 1px solid #232D44; margin-bottom: 14px; }
.eyebrow { font-size: 11.5px; letter-spacing: .12em; text-transform: uppercase; color: #4C9AFF; font-weight: 600; }
.hero h1 { font-size: 28px; font-weight: 700; color: #F2F5FA; margin: 4px 0 4px; letter-spacing: -.01em; line-height: 1.2; }
.hero p { font-size: 14px; color: #8892A6; margin: 0; max-width: 780px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { font-size: 12px; color: #C3CBDA; background: #121826; border: 1px solid #232D44; border-radius: 999px; padding: 4px 11px; white-space: nowrap; }
.chip b { color: #F2F5FA; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #232D44; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; color: #8892A6; font-weight: 500; font-size: 14px; }
.stTabs [aria-selected="true"] { color: #F2F5FA; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #4C9AFF; }

.sec-title { font-size: 16px; font-weight: 600; color: #F2F5FA; margin: 18px 0 2px; }
.sec-sub { font-size: 13px; color: #8892A6; margin-bottom: 12px; max-width: 900px; line-height: 1.55; }
.note { background: #121826; border: 1px solid #232D44; border-left: 3px solid #4C9AFF; border-radius: 8px;
        padding: 12px 16px; font-size: 13.5px; color: #C3CBDA; line-height: 1.6; margin: 8px 0 14px; }
.note b { color: #F2F5FA; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 6px 0 14px; }
.kpi { background: #121826; border: 1px solid #232D44; border-radius: 10px; padding: 14px 16px; }
.kpi-label { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #8892A6; font-weight: 600; }
.kpi-value { font-size: 26px; font-weight: 700; color: #F2F5FA; margin-top: 8px; display: flex; align-items: center; gap: 10px; line-height: 1.15; }
.kpi-sub { font-size: 12px; color: #6F7A90; margin-top: 5px; }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.flag { width: 18px; height: 13px; object-fit: cover; border-radius: 2px; flex: none; box-shadow: 0 0 0 1px rgba(255,255,255,.14); }
.flag.lg { width: 34px; height: 24px; border-radius: 3px; }
.flag.xl { width: 56px; height: 40px; border-radius: 4px; }

.sandbox-bar { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;
               background: #121826; border: 1px solid #232D44; border-left: 3px solid #A78BFA; border-radius: 10px;
               padding: 10px 16px; }
.sandbox-bar .sb-text { font-size: 13px; color: #C3CBDA; line-height: 1.5; }
.sandbox-bar .sb-text b { color: #F2F5FA; font-weight: 600; }
.sandbox-bar .sb-count { display: inline-block; margin-left: 8px; font-size: 12px; font-weight: 600; color: #D8CCFF;
                         background: rgba(167,139,250,.16); border: 1px solid rgba(167,139,250,.4);
                         border-radius: 999px; padding: 2px 10px; white-space: nowrap; }
.sandbox-bar .sb-count.zero { color: #8892A6; background: transparent; border-color: #232D44; }

.vs-head { display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 18px; background: #121826;
           border: 1px solid #232D44; border-radius: 12px; padding: 18px 22px; margin: 8px 0 14px; }
.vs-team { display: flex; align-items: center; gap: 14px; }
.vs-team.right { flex-direction: row-reverse; text-align: right; }
.vs-name { font-size: 22px; font-weight: 700; color: #F2F5FA; line-height: 1.15; }
.vs-meta { font-size: 12.5px; color: #8892A6; margin-top: 3px; }
.vs-mid { text-align: center; }
.vs-mid .vs { font-size: 12px; letter-spacing: .14em; color: #6F7A90; font-weight: 600; }
.vs-mid .rd { font-size: 12px; color: #8892A6; margin-top: 4px; }
.prob-card { background: #121826; border: 1px solid #232D44; border-radius: 10px; padding: 14px 16px; }
.prob-title { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #8892A6; font-weight: 600; }
.prob-bar { display: flex; height: 36px; border-radius: 6px; overflow: hidden; margin-top: 10px; }
.seg { display: flex; align-items: center; padding: 0 11px; font-weight: 600; font-size: 14px; color: #0B0F17;
       white-space: nowrap; font-variant-numeric: tabular-nums; }
.seg.a { background: #4C9AFF; }
.seg.b { background: #F0A35C; justify-content: flex-end; }
.prob-foot { font-size: 12px; color: #6F7A90; margin-top: 8px; }

.stat-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.stat-table th { text-align: right; font-size: 11px; letter-spacing: .07em; text-transform: uppercase; color: #8892A6;
                 font-weight: 600; padding: 8px 12px; border-bottom: 1px solid #232D44; }
.stat-table th:first-child, .stat-table td:first-child { text-align: left; }
.stat-table td { text-align: right; padding: 9px 12px; border-bottom: 1px solid #1A2335; color: #AEB7C9; font-variant-numeric: tabular-nums; }
.stat-table td:first-child { color: #C3CBDA; }
.stat-table td.best { color: #F2F5FA; font-weight: 600; }
.stat-table tr:hover td { background: #121826; }
.stat-wrap { background: #0F1522; border: 1px solid #232D44; border-radius: 10px; padding: 4px 6px 8px; }

.match-pill { display: inline-block; font-size: 12px; color: #C3CBDA; background: #121826; border: 1px solid #232D44;
              border-radius: 999px; padding: 3px 10px; margin: 0 6px 6px 0; }
.match-pill b { color: #F2F5FA; font-weight: 600; }
.small-muted { font-size: 12px; color: #6F7A90; }
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Drabinka renderowana jest w iframe komponentu (src/bracket_component), więc ma własny arkusz stylów.
BRACKET_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; }
body { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; color: #E6EAF2; }
.flag { width: 18px; height: 13px; object-fit: cover; border-radius: 2px; flex: none; box-shadow: 0 0 0 1px rgba(255,255,255,.14); }
.flag.xl { width: 56px; height: 40px; border-radius: 4px; }

.bracket-scroll { overflow-x: auto; padding: 4px 2px 10px; }
.bracket { display: flex; gap: 22px; min-width: 1240px; }
.bcol { flex: 1 1 0; min-width: 182px; display: flex; flex-direction: column; }
.bcol-title { font-size: 11px; letter-spacing: .09em; text-transform: uppercase; color: #8892A6; font-weight: 600;
              padding-bottom: 7px; margin-bottom: 8px; border-bottom: 1px solid #232D44; }
.bcol-body { display: flex; flex-direction: column; }
.slot { flex: 1 1 0; position: relative; display: flex; align-items: center; padding: 3px 0; }
.slot > div { width: 100%; }
.slot.top::after, .slot.bot::after, .slot.single::after { content: ""; position: absolute; right: -11px; width: 11px;
    border: 0 solid #2B3753; }
.slot.top::after { top: 50%; height: 50%; border-top-width: 1px; border-right-width: 1px; }
.slot.bot::after { bottom: 50%; height: 50%; border-bottom-width: 1px; border-right-width: 1px; }
.slot.single::after { top: 50%; height: 0; border-top-width: 1px; }
.slot.in::before { content: ""; position: absolute; left: -11px; top: 50%; width: 11px; border-top: 1px solid #2B3753; }

.mcard { --st: #4C9AFF; background: #121826; border: 1px solid #232D44; border-left: 3px solid var(--st);
         border-radius: 8px; overflow: hidden; font-size: 12px; transition: border-color .15s, background .15s; }
.mcard:hover { background: #161E31; border-color: #34425F; border-left-color: var(--st); }
.mcard.real { --st: #E3B341; }
.mcard.ai { --st: #4C9AFF; }
.mcard.user { --st: #A78BFA; }
.mhead { padding: 4px 8px 1px; font-size: 9px; letter-spacing: .08em; text-transform: uppercase; color: var(--st); font-weight: 600; }
.mrow { display: flex; align-items: center; gap: 7px; padding: 3px 8px; min-height: 22px; color: #7C879C;
        transition: background .12s, color .12s; }
.mrow.win { color: #F2F5FA; font-weight: 600; background: rgba(255,255,255,.045); }
.mcard.user .mrow.win { background: rgba(167,139,250,.18); }
.mname { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mprob { font-variant-numeric: tabular-nums; font-size: 11.5px; flex: none; }
.upset { color: #FF8A3D; font-size: 9px; flex: none; cursor: help; }
.mfoot { padding: 2px 8px 5px; font-size: 10px; color: #6F7A90; font-variant-numeric: tabular-nums; }

/* Tryb Sandbox: wiersze drużyn są klikalne */
.interactive .mrow[data-team] { cursor: pointer; user-select: none; }
.interactive .mrow[data-team]:hover { background: rgba(167,139,250,.14); color: #F2F5FA; }
.interactive .mcard.user .mrow.win:hover { background: rgba(167,139,250,.28); }
.interactive .mrow.pending { opacity: .55; background: rgba(167,139,250,.22); cursor: progress; }

.champ-card { text-align: center; padding: 18px 10px 16px; border: 1px solid #4A3F1C; border-radius: 10px;
              background: linear-gradient(180deg, #1C1A12 0%, #121826 100%); }
.champ-label { font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase; color: #E3B341; font-weight: 600; }
.champ-name { font-size: 20px; font-weight: 700; color: #F2F5FA; margin-top: 10px; }
.champ-sub { font-size: 12px; color: #8892A6; margin-top: 4px; }
"""

_bracket_component = components.declare_component(
    "wc_bracket", path=str(Path(__file__).resolve().parent / "src" / "bracket_component")
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#C3CBDA", size=13),
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(bgcolor="#161E31", bordercolor="#34425F", font=dict(color="#F2F5FA")),
)


# --------------------------------------------------------------------------- #
# Dane i model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def build_database_if_missing() -> bool:
    """Jednorazowa samonaprawa: gdy w kontenerze/na hostingu brak bazy, buduje ją pipeline'em (~20 s)."""
    if DB_PATH.exists():
        return True
    proc = subprocess.run(
        [sys.executable, "-m", "src.run_pipeline"],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode == 0 and DB_PATH.exists()


if not DB_PATH.exists():
    with st.spinner("Brak bazy danych - buduję ją z surowych danych (jednorazowo, ok. 20 s)..."):
        build_database_if_missing.clear()
        built = build_database_if_missing()
    if not built:
        st.error(
            "Nie udało się zbudować bazy `data/world_cup.duckdb`. Uruchom `python -m src.run_pipeline`, "
            "aby zobaczyć szczegóły błędu, a następnie odśwież stronę."
        )
        st.stop()


@st.cache_resource(show_spinner="Trenowanie modelu hybrydowego (jednorazowo)...")
def load_predictor() -> HybridMatchPredictor:
    con = get_connection(read_only=True)
    try:
        return HybridMatchPredictor(con)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_ratings(mtime: float) -> pd.DataFrame:
    return pd.read_csv(CURRENT_ELO_CSV)


@st.cache_data(show_spinner=False)
def load_saved_monte_carlo(mtime: float) -> pd.DataFrame:
    raw = pd.read_csv(MONTE_CARLO_CSV)
    return pd.DataFrame(
        {
            "team": raw["Team"],
            "reach_round_of_16": raw["Reach_R16_Prob"],
            "reach_quarter_final": raw["Reach_Quarter_Prob"],
            "reach_semi_final": raw["Reach_Semi_Prob"],
            "reach_final": raw["Reach_Final_Prob"],
            "win_tournament": raw["Win_Probability"],
        }
    ).sort_values("win_tournament", ascending=False, ignore_index=True)


def file_mtime(path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


predictor_base = load_predictor()
WC_TEAMS = set(predictor_base.style_table.index)


# --------------------------------------------------------------------------- #
# Stan sesji
# --------------------------------------------------------------------------- #
st.session_state.setdefault("form_mult", 1.0)
st.session_state.setdefault("style_mult", 1.0)
# Ręczne wybory Sandbox: {frozenset({drużyna A, drużyna B}): zwycięzca}
st.session_state.setdefault("sandbox_overrides", {})


def reset_calibration() -> None:
    st.session_state["form_mult"] = 1.0
    st.session_state["style_mult"] = 1.0


def clear_sandbox() -> None:
    st.session_state["sandbox_overrides"] = {}


# --------------------------------------------------------------------------- #
# Pasek boczny
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="side-brand">Mistrzostwa Świata 2026</div>'
        '<div class="side-sub">Analityka fazy pucharowej</div>'
        '<div class="side-title">Panel sterowania symulacją</div>',
        unsafe_allow_html=True,
    )
    mode_label = st.radio("Tryb symulacji", list(MODE_LABELS.values()), key="mode_label", help=MODE_HELP)
    mode = next(k for k, v in MODE_LABELS.items() if v == mode_label)

    st.markdown('<div class="side-title" style="margin-top:22px">Kalibracja modelu hybrydowego</div>', unsafe_allow_html=True)
    st.slider(
        "Waga aktualnej formy",
        0.0, 2.0, step=0.05, key="form_mult", format="%.2fx",
        help="Siła korekty wynikającej z formy z ostatnich 8 meczów. 0.0x wyłącza formę, 1.0x to ustawienie bazowe.",
    )
    st.slider(
        "Waga stylu gry (FBref)",
        0.0, 2.0, step=0.05, key="style_mult", format="%.2fx",
        help="Siła korekty wynikającej ze stylu gry w fazie grupowej: posiadania piłki, goli na 90 minut i skuteczności.",
    )
    st.button("Przywróć ustawienia domyślne", on_click=reset_calibration, **STRETCH)

    st.markdown('<div class="side-title" style="margin-top:22px">Legenda</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="legend-row"><span class="legend-dot" style="background:#E3B341"></span>Wynik rzeczywisty</div>'
        '<div class="legend-row"><span class="legend-dot" style="background:#4C9AFF"></span>Predykcja AI</div>'
        '<div class="legend-row"><span class="legend-dot" style="background:#A78BFA"></span>Wybór użytkownika</div>'
        '<div class="legend-row"><span style="color:#FF8A3D;width:10px;text-align:center;font-size:10px">&#9650;</span>'
        "Niespodzianka: awans zespołu z szansą poniżej 50%</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="side-note" style="margin-top:18px">Prawdopodobieństwa łączą siłę bazową drużyn (rating Elo), '
        "formę z ostatnich 8 meczów, styl gry z fazy grupowej oraz bonus gospodarza "
        f"(+{HOST_BONUS:.0f} pkt Elo: USA w każdej rundzie, Meksyk i Kanada do 1/8 finału).</div>",
        unsafe_allow_html=True,
    )

form_mult = float(st.session_state["form_mult"])
style_mult = float(st.session_state["style_mult"])
predictor = predictor_base.with_multipliers(form_mult, style_mult)


# --------------------------------------------------------------------------- #
# Narzędzia prezentacji
# --------------------------------------------------------------------------- #
def esc(text) -> str:
    return html.escape(str(text), quote=True)


def flag_img(team: str, size: str = "") -> str:
    url = flag_url(team, 80 if size else 40)
    cls = f"flag {size}".strip()
    return f'<img class="{cls}" src="{url}" alt="" loading="lazy">' if url else ""


def team_label(team: str) -> str:
    """Etykieta tekstowa do kontrolek (lista rozwijana)."""
    emoji = flag_emoji(team)
    return f"{emoji} {team}".strip()


def signed(value: float, digits: int = 1) -> str:
    value = round(float(value), digits) + 0.0  # eliminuje "-0.0"
    return f"{value:+.{digits}f}"


def section(title: str, subtitle: str = "") -> None:
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="sec-title">{title}</div>{sub}', unsafe_allow_html=True)


def apply_plot_style(fig: go.Figure, **extra) -> go.Figure:
    fig.update_layout(**{**PLOT_LAYOUT, **extra})
    fig.update_xaxes(gridcolor=COL_GRID, zerolinecolor=COL_GRID, linecolor=COL_GRID)
    fig.update_yaxes(gridcolor=COL_GRID, zerolinecolor=COL_GRID, linecolor=COL_GRID)
    return fig


def pct_rank(reference: np.ndarray, value: float, higher_is_better: bool = True) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[~np.isnan(reference)]
    if len(reference) == 0 or value != value:
        return 50.0
    share = np.mean(reference <= value) if higher_is_better else np.mean(reference >= value)
    return float(share * 100.0)


# --------------------------------------------------------------------------- #
# Drabinka: komponenty HTML
# (każdy fragment składany bez wcięć i znaków nowej linii - inaczej parser Markdown
#  traktuje go jak blok kodu)
# --------------------------------------------------------------------------- #
def match_status(match: dict) -> Tuple[str, str]:
    if match["is_real"]:
        return "real", "Wynik rzeczywisty"
    if match["is_override"]:
        return "user", "Wybór użytkownika"
    return "ai", "Predykcja AI"


def match_tooltip(m: dict) -> str:
    a, b = m["team_a"], m["team_b"]
    lines = [
        f"{a} vs {b} ({m['round']})",
        f"Rating Elo: {m['elo_a']:.0f} vs {m['elo_b']:.0f}",
        f"Czyste Elo: {m['elo_prob_a']:.1f}% - {100 - m['elo_prob_a']:.1f}%",
        f"Model hybrydowy: {m['prob_a']:.1f}% - {m['prob_b']:.1f}%",
        f"Korekty dla {a} (pp): gospodarz {signed(m['host_impact'])}, kalibracja {signed(m['calibration_impact'])}, "
        f"forma {signed(m['form_impact'])}, styl {signed(m['style_impact'])}",
    ]
    return "&#10;".join(esc(line) for line in lines)


def match_card(m: dict, interactive: bool = False) -> str:
    a, b, winner = m["team_a"], m["team_b"], m["winner"]
    status_cls, status_text = match_status(m)
    sign = 1.0 if winner == a else -1.0
    mini = f"Forma {signed(sign * m['form_impact'])} &middot; Styl {signed(sign * m['style_impact'])} pp"
    upset_hint = esc(f"Niespodzianka: {winner} awansuje mimo {m['winner_prob']:.1f}% szans wg modelu")

    def row(team: str, prob: float) -> str:
        is_win = team == winner
        marker = f'<span class="upset" title="{upset_hint}">&#9650;</span>' if (is_win and m["is_upset"]) else ""
        attrs = ""
        if interactive:
            attrs = f' data-a="{esc(a)}" data-b="{esc(b)}" data-team="{esc(team)}"'
            if is_win and m["is_override"]:
                attrs += f' title="{esc("Kliknij ponownie, aby wrócić do predykcji AI")}"'
        return (
            f'<div class="mrow{" win" if is_win else ""}"{attrs}>{flag_img(team)}'
            f'<span class="mname">{esc(team)}</span>{marker}<span class="mprob">{prob:.1f}%</span></div>'
        )

    return (
        f'<div class="mcard {status_cls}" title="{match_tooltip(m)}">'
        f'<div class="mhead">{status_text}</div>'
        f"{row(a, m['prob_a'])}{row(b, m['prob_b'])}"
        f'<div class="mfoot">{mini}</div></div>'
    )


def bracket_html(result: dict, interactive: bool = False) -> str:
    rounds = result["rounds"]
    height = len(rounds[0]) * SLOT_HEIGHT
    columns: List[str] = []
    for r, (label, matches) in enumerate(zip(ROUND_LABELS, rounds)):
        slots = []
        for j, m in enumerate(matches):
            if r == len(rounds) - 1:
                pos = "single"
            else:
                pos = "top" if j % 2 == 0 else "bot"
            cls = f"slot {pos}" + (" in" if r > 0 else "")
            slots.append(f'<div class="{cls}">{match_card(m, interactive)}</div>')
        columns.append(
            f'<div class="bcol"><div class="bcol-title">{label}</div>'
            f'<div class="bcol-body" style="height:{height}px">{"".join(slots)}</div></div>'
        )

    final = rounds[-1][0]
    champ = result["champion"]
    runner_up = final["team_b"] if final["winner"] == final["team_a"] else final["team_a"]
    columns.append(
        '<div class="bcol"><div class="bcol-title">Mistrz Świata</div>'
        f'<div class="bcol-body" style="height:{height}px"><div class="slot in">'
        '<div class="champ-card"><div class="champ-label">Mistrz Świata 2026</div>'
        f'<div style="margin-top:12px">{flag_img(champ, "xl")}</div>'
        f'<div class="champ-name">{esc(champ)}</div>'
        f'<div class="champ-sub">Finał z {esc(runner_up)} &middot; {final["winner_prob"]:.1f}%</div>'
        "</div></div></div></div>"
    )
    return f'<div class="bracket-scroll"><div class="bracket">{"".join(columns)}</div></div>'


def kpi_html(items: List[Tuple[str, str, str]]) -> str:
    cards = "".join(
        f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
        for label, value, sub in items
    )
    return f'<div class="kpi-grid">{cards}</div>'


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
def toggle_sandbox_pick(a: str, b: str, team: str) -> None:
    """Obsługa kliknięcia w drużynę: ustawia zwycięzcę meczu a-b albo cofa wybór (ponowne kliknięcie)."""
    overrides: Dict[frozenset, str] = dict(st.session_state["sandbox_overrides"])
    pair = frozenset({a, b})
    if len(pair) != 2 or team not in pair:
        return
    if overrides.get(pair) == team:
        del overrides[pair]  # ponowne kliknięcie w wybraną drużynę -> powrót do predykcji AI
    else:
        overrides[pair] = team
    # zmiana zwycięzcy we wcześniejszej rundzie unieważnia ręczne wybory dla starej ścieżki
    st.session_state["sandbox_overrides"], _ = prune_stale_overrides(
        overrides, predictor=predictor_base, form_multiplier=form_mult, style_multiplier=style_mult
    )


def sandbox_bar(n_changes: int) -> str:
    count_cls = "sb-count" + ("" if n_changes else " zero")
    return (
        '<div class="sandbox-bar"><div class="sb-text">'
        "<b>Tryb Sandbox.</b> Kliknij drużynę w dowolnym meczu drabinki (od 1/16 finału do Finału), aby ją awansować. "
        "Ponowne kliknięcie wybranej (fioletowej) drużyny przywraca predykcję AI."
        f'<span class="{count_cls}">Ręczne zmiany: {n_changes}</span>'
        "</div></div>"
    )


# --------------------------------------------------------------------------- #
# Nagłówek
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="hero"><div>'
    '<div class="eyebrow">FIFA World Cup 2026 &middot; Faza pucharowa</div>'
    "<h1>Prognozy, szanse i scenariusze</h1>"
    "<p>Model hybrydowy łączy rating Elo, aktualną formę i styl gry drużyn. "
    "Sprawdź przewidywany przebieg turnieju, porównaj dowolne drużyny i zbadaj wrażliwość modelu.</p>"
    "</div>"
    '<div class="chips">'
    f'<span class="chip">Tryb: <b>{esc(MODE_LABELS[mode])}</b></span>'
    f'<span class="chip">Waga formy: <b>{form_mult:.2f}x</b></span>'
    f'<span class="chip">Waga stylu: <b>{style_mult:.2f}x</b></span>'
    "</div></div>",
    unsafe_allow_html=True,
)

tab_bracket, tab_h2h, tab_mc, tab_rank = st.tabs(
    [
        "Drabinka Fazy Pucharowej",
        "Analiza Meczowa Head-to-Head (Elo + Forma + Styl)",
        "Symulacja Monte Carlo (10 000 turniejów)",
        "Światowy Ranking Elo (211 drużyn FIFA)",
    ]
)

# =========================================================================== #
# Zakładka 1 - drabinka
# =========================================================================== #
with tab_bracket:
    kpi_slot = st.container()
    sandbox_slot = st.container()

    overrides: Dict[frozenset, str] = {}
    if mode == "sandbox":
        # czyścimy nieaktualne wybory (np. po zmianie wag formy/stylu mogły zmienić się pary kolejnych rund)
        overrides, result = prune_stale_overrides(
            st.session_state["sandbox_overrides"],
            predictor=predictor_base, form_multiplier=form_mult, style_multiplier=style_mult,
        )
        st.session_state["sandbox_overrides"] = overrides
        with sandbox_slot:
            bar_col, clear_col = st.columns([5, 1.2], vertical_alignment="center")
            bar_col.markdown(sandbox_bar(len(overrides)), unsafe_allow_html=True)
            clear_col.button(
                "Wyczyść własny scenariusz", on_click=clear_sandbox, key="sb_clear",
                disabled=not overrides, **STRETCH,
            )
    else:
        result = simulate_bracket(
            mode, None, predictor=predictor_base, form_multiplier=form_mult, style_multiplier=style_mult
        )
    all_matches = [m for rnd in result["rounds"] for m in rnd]
    final_match = result["rounds"][-1][0]
    champion = result["champion"]

    n_real = sum(1 for m in all_matches if m["is_real"])
    n_user = sum(1 for m in all_matches if m["is_override"])
    if mode == "actual_plus_ai":
        upset_sub = f"spośród {n_real} rozegranych meczów"
    elif mode == "sandbox":
        upset_sub = f"w tym {n_user} meczów wybranych ręcznie" if n_user else "scenariusz bez ręcznych wyborów"
    else:
        upset_sub = "tryb 100% AI - zawsze wygrywa faworyt"
    confidence = float(np.mean([max(m["prob_a"], m["prob_b"]) for m in all_matches]))

    with kpi_slot:
        st.markdown(
            kpi_html(
                [
                    (
                        "Przewidywany Mistrz Świata",
                        f"{flag_img(champion, 'lg')}<span>{esc(champion)}</span>",
                        f"Finał: {esc(final_match['team_a'])} vs {esc(final_match['team_b'])}",
                    ),
                    (
                        "Szansa Mistrza w Finale",
                        f"{final_match['winner_prob']:.1f}%",
                        "szansa zwycięzcy w meczu finałowym wg modelu",
                    ),
                    ("Liczba niespodzianek w turnieju", str(result["upset_count"]), upset_sub),
                    (
                        "Średnia pewność modelu",
                        f"{confidence:.1f}%",
                        f"średnia szansa faworyta w {len(all_matches)} meczach",
                    ),
                ]
            ),
            unsafe_allow_html=True,
        )

    section(
        "Drabinka turniejowa",
        "Od 1/16 finału do Finału. Najedź kursorem na mecz, aby zobaczyć rozbicie prawdopodobieństwa: "
        "czyste Elo, bonus gospodarza oraz korekty z formy i stylu gry."
        + (" W trybie Sandbox kliknij drużynę, aby ustawić zwycięzcę meczu." if mode == "sandbox" else ""),
    )
    click = _bracket_component(
        html=bracket_html(result, interactive=(mode == "sandbox")),
        css=BRACKET_CSS,
        interactive=(mode == "sandbox"),
        key="bracket_component",
        default=None,
    )
    # każde kliknięcie ma unikalne id - przetwarzamy je dokładnie raz (wartość komponentu przetrwa rerun)
    if click and click.get("id") != st.session_state.get("_bracket_last_click"):
        st.session_state["_bracket_last_click"] = click["id"]
        if mode == "sandbox":
            toggle_sandbox_pick(click["a"], click["b"], click["team"])
            st.rerun()

    with st.expander("Metodologia i ograniczenia"):
        st.markdown(
            f"""
- **Siła bazowa.** Rating Elo liczony na historii meczów reprezentacji od 1980 roku, aktualizowany po każdym spotkaniu.
- **Forma.** Średnia zmiana ratingu oraz bilans bramek z 8 ostatnich meczów. Wpływ formy jest uczony na danych historycznych.
- **Styl gry.** Wskaźnik złożony z posiadania piłki, goli (bez rzutów karnych) na 90 minut i skuteczności ponad poziom
  wynikający z posiadania. Dane pochodzą z fazy grupowej MŚ 2026 (FBref).
- **Bonus gospodarza.** +{HOST_BONUS:.0f} pkt Elo. Od ćwierćfinałów wszystkie mecze odbywają się w USA, dlatego Meksyk
  i Kanada mają bonus tylko w 1/16 i 1/8 finału, a USA w każdej rundzie.
- **Drabinka deterministyczna.** W każdym meczu awansuje faworyt modelu. Rating i forma aktualizują się po każdej
  rundzie, więc kolejne pojedynki uwzględniają wcześniejsze wyniki.
- **Niespodzianka.** Mecz, w którym awansował zespół z szansą poniżej 50% według modelu.

Prognozy mają charakter analityczny i nie stanowią porady bukmacherskiej.
"""
        )

# =========================================================================== #
# Zakładka 2 - Head-to-Head
# =========================================================================== #
STAGE_LABELS = dict(zip(STAGES, ROUND_LABELS))


def style_row(team: str) -> Optional[pd.Series]:
    return predictor.style_table.loc[team] if team in predictor.style_table.index else None


def commentary(v: dict) -> str:
    a, b = esc(v["team_a"]), esc(v["team_b"])
    pa, pb = v["prob_a"], 100.0 - v["prob_a"]
    fav = a if pa >= 50 else b
    parts = [f"Model hybrydowy wskazuje na <b>{fav}</b> ({max(pa, pb):.1f}%)."]
    if (v["elo_prob_a"] >= 50) != (pa >= 50):
        parts.append("Korekty odwracają faworyta względem samego Elo.")
    drivers = {
        "bonus gospodarza": v["host_impact"],
        "rekalibracja modelu": v["calibration_impact"],
        "aktualna forma": v["form_impact"],
        "styl gry": v["style_impact"],
    }
    name, val = max(drivers.items(), key=lambda kv: abs(kv[1]))
    if abs(val) >= 0.05:
        beneficiary = a if val > 0 else b
        parts.append(f"Największa korekta pochodzi z czynnika: {name} ({abs(val):.1f} pp na korzyść {beneficiary}).")
    else:
        parts.append("Korekty są pomijalne - ocena zbliżona do samego Elo.")
    return " ".join(parts)


def prob_bar(title: str, team_a: str, team_b: str, pa: float, foot: str = "") -> str:
    pb = 100.0 - pa
    return (
        f'<div class="prob-card"><div class="prob-title">{title}</div>'
        f'<div class="prob-bar"><div class="seg a" style="width:{pa:.2f}%">{pa:.1f}%</div>'
        f'<div class="seg b" style="width:{pb:.2f}%">{pb:.1f}%</div></div>'
        f'<div class="prob-foot">{foot}</div></div>'
    )


def waterfall_figure(v: dict) -> go.Figure:
    a = v["team_a"]
    steps = [
        ("Elo bez bonusu<br>gospodarza", v["elo_neutral_prob_a"], "absolute"),
        ("Bonus<br>gospodarza", v["host_impact"], "relative"),
        ("Rekalibracja<br>modelu", v["calibration_impact"], "relative"),
        ("Aktualna forma<br>(8 meczów)", v["form_impact"], "relative"),
        ("Styl gry<br>(FBref)", v["style_impact"], "relative"),
        ("Model<br>hybrydowy", v["prob_a"], "total"),
    ]
    running, cumulative = 0.0, []
    for _, value, kind in steps:
        running = value if kind == "absolute" else (running + value if kind == "relative" else value)
        cumulative.append(running)
    texts = [f"{steps[0][1]:.1f}%"] + [f"{signed(s[1], 1)} pp" for s in steps[1:-1]] + [f"{steps[-1][1]:.1f}%"]

    fig = go.Figure(
        go.Waterfall(
            x=[s[0] for s in steps],
            y=[s[1] for s in steps],
            measure=[s[2] for s in steps],
            text=texts,
            textposition="outside",
            connector=dict(line=dict(color="#34425F", width=1)),
            increasing=dict(marker=dict(color=COL_GREEN)),
            decreasing=dict(marker=dict(color=COL_RED)),
            totals=dict(marker=dict(color=COL_BLUE)),
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        )
    )
    lo = max(0.0, min(cumulative) - 12.0)
    hi = min(100.0, max(cumulative) + 12.0)
    apply_plot_style(
        fig,
        title=dict(text=f"Szansa awansu: {a} - dekompozycja czynników", font=dict(size=15, color=COL_TEXT), x=0.0),
        showlegend=False,
        height=420,
        yaxis=dict(range=[lo, hi], ticksuffix="%", title=None),
        xaxis=dict(title=None),
    )
    return fig


def radar_figure(v: dict) -> go.Figure:
    a, b = v["team_a"], v["team_b"]
    states = [predictor.get_team_state(t) for t in WC_TEAMS]
    ref = {
        "elo": np.array([s["elo"] for s in states]),
        "form": np.array([s["form"] for s in states]),
        "gf": np.array([s["goals_scored"] for s in states]),
        "ga": np.array([s["goals_conceded"] for s in states]),
    }
    sty = predictor.style_table
    sa, sb = style_row(a), style_row(b)

    axes = ["Siła (Elo)", "Forma", "Atak", "Obrona"]

    def vals(prefix: str) -> List[float]:
        return [
            pct_rank(ref["elo"], v[f"elo_{prefix}"]),
            pct_rank(ref["form"], v[f"form_{prefix}"]),
            pct_rank(ref["gf"], v[f"gf_{prefix}"]),
            pct_rank(ref["ga"], v[f"ga_{prefix}"], higher_is_better=False),
        ]

    va, vb = vals("a"), vals("b")
    if sa is not None and sb is not None:
        axes += ["Posiadanie piłki", "Skuteczność"]
        va += [pct_rank(sty["possession"].to_numpy(), sa["possession"]),
               pct_rank(sty["clinical_vs_possession"].to_numpy(), sa["clinical_vs_possession"])]
        vb += [pct_rank(sty["possession"].to_numpy(), sb["possession"]),
               pct_rank(sty["clinical_vs_possession"].to_numpy(), sb["clinical_vs_possession"])]

    fig = go.Figure()
    for name, values, color, fill in ((a, va, COL_BLUE, "rgba(76,154,255,.22)"), (b, vb, COL_ORANGE, "rgba(240,163,92,.20)")):
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1], theta=axes + axes[:1], name=name, fill="toself",
                line=dict(color=color, width=2), fillcolor=fill,
                hovertemplate="%{theta}: %{r:.0f}. percentyl<extra>" + name + "</extra>",
            )
        )
    apply_plot_style(
        fig,
        title=dict(text="Profil drużyn na tle 48 uczestników MŚ (percentyl)", font=dict(size=15, color=COL_TEXT), x=0.0),
        height=420,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], gridcolor=COL_GRID, linecolor=COL_GRID, tickfont=dict(size=10, color=COL_MUTED)),
            angularaxis=dict(gridcolor=COL_GRID, linecolor=COL_GRID),
        ),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
        margin=dict(l=40, r=40, t=50, b=30),
    )
    return fig


def stats_table_html(v: dict) -> str:
    a, b = v["team_a"], v["team_b"]
    sa, sb = predictor.get_team_state(a), predictor.get_team_state(b)
    ya, yb = style_row(a), style_row(b)
    nan = float("nan")

    def fmt(x: float, f: str) -> str:
        return "brak danych" if x != x else format(x, f)

    rows: List[Tuple[str, float, float, str, int]] = [
        ("Rating Elo", v["elo_a"], v["elo_b"], ".0f", 1),
        ("Zmiana formy (Elo na mecz, ostatnie 8)", v["form_a"], v["form_b"], "+.2f", 1),
        ("Bramki strzelone na mecz (ostatnie 8)", v["gf_a"], v["gf_b"], ".2f", 1),
        ("Bramki stracone na mecz (ostatnie 8)", v["ga_a"], v["ga_b"], ".2f", -1),
        ("Posiadanie piłki, % (faza grupowa)", float(sa["possession"]), float(sb["possession"]), ".1f", 1),
        ("Gole na 90 minut (faza grupowa)", float(sa["goals_per90"]), float(sb["goals_per90"]), ".2f", 1),
        ("Gole bez karnych na 90 minut", float(ya["npg_per90"]) if ya is not None else nan,
         float(yb["npg_per90"]) if yb is not None else nan, ".2f", 1),
        ("Skuteczność ponad poziom posiadania (gole na 90 min)",
         float(ya["clinical_vs_possession"]) if ya is not None else nan,
         float(yb["clinical_vs_possession"]) if yb is not None else nan, "+.2f", 1),
        ("Indeks stylu gry (odchylenia std.)", float(sa["style_score"]) if ya is not None else nan,
         float(sb["style_score"]) if yb is not None else nan, "+.2f", 1),
    ]
    body = []
    for label, xa, xb, f, direction in rows:
        best_a = best_b = False
        if xa == xa and xb == xb and xa != xb:
            best_a = (xa > xb) == (direction > 0)
            best_b = not best_a
        body.append(
            f"<tr><td>{label}</td><td class=\"{'best' if best_a else ''}\">{fmt(xa, f)}</td>"
            f"<td class=\"{'best' if best_b else ''}\">{fmt(xb, f)}</td></tr>"
        )
    head = (
        f"<tr><th>Statystyka</th><th>{flag_img(a)} {esc(a)}</th><th>{flag_img(b)} {esc(b)}</th></tr>"
    )
    return f'<div class="stat-wrap"><table class="stat-table">{head}{"".join(body)}</table></div>'


with tab_h2h:
    section(
        "Analiza meczowa",
        "Wybierz mecz z drabinki lub dowolne dwie reprezentacje. Zobacz, jak bazowe prawdopodobieństwo Elo "
        "zmienia się pod wpływem bonusu gospodarza, aktualnej formy i stylu gry.",
    )

    source = st.radio(
        "Źródło pojedynku",
        ["Mecz z drabinki", "Dowolne dwie reprezentacje"],
        horizontal=True,
        key="h2h_source",
        label_visibility="collapsed",
    )

    view: Optional[dict] = None
    note = ""
    if source == "Mecz z drabinki":
        idx = st.selectbox(
            "Mecz",
            range(len(all_matches)),
            format_func=lambda i: (
                f"{all_matches[i]['round']}  |  {team_label(all_matches[i]['team_a'])} vs "
                f"{team_label(all_matches[i]['team_b'])}"
            ),
            key="h2h_match",
        )
        view = all_matches[idx]
        note = (
            "Rating Elo i forma uwzględniają wyniki wcześniejszych rund w wybranym trybie symulacji. "
            "Statystyki stylu gry pochodzą z fazy grupowej."
        )
    else:
        ratings_all = load_ratings(file_mtime(CURRENT_ELO_CSV))
        team_options = list(ratings_all.sort_values("current_elo", ascending=False)["team"])
        c1, c2, c3 = st.columns([2, 2, 2])
        team_a = c1.selectbox("Drużyna A", team_options, index=team_options.index("Argentina"),
                              format_func=team_label, key="h2h_a")
        team_b = c2.selectbox("Drużyna B", team_options, index=team_options.index("France"),
                              format_func=team_label, key="h2h_b")
        stage = c3.selectbox(
            "Runda meczu", list(STAGES), format_func=lambda s: STAGE_LABELS[s], key="h2h_stage",
            help="Decyduje o bonusie gospodarza: USA mają go w każdej rundzie, Meksyk i Kanada tylko w 1/16 i 1/8 finału.",
        )
        if team_a == team_b:
            st.warning("Wybierz dwie różne reprezentacje.")
        else:
            view = predictor.predict_match(team_a, team_b, stage=stage)
            view.update(round=STAGE_LABELS[stage])
            note = (
                "Stan drużyn z końca fazy grupowej (przed pierwszym meczem pucharowym). "
                "Dla reprezentacji, które nie grają na MŚ 2026, brak statystyk stylu gry z FBref."
            )

    if view is not None:
        a, b = view["team_a"], view["team_b"]
        st.markdown(
            '<div class="vs-head">'
            f'<div class="vs-team">{flag_img(a, "xl")}<div><div class="vs-name">{esc(a)}</div>'
            f'<div class="vs-meta">Elo {view["elo_a"]:.0f}</div></div></div>'
            f'<div class="vs-mid"><div class="vs">VS</div><div class="rd">{esc(view.get("round", ""))}</div></div>'
            f'<div class="vs-team right">{flag_img(b, "xl")}<div><div class="vs-name">{esc(b)}</div>'
            f'<div class="vs-meta">Elo {view["elo_b"]:.0f}</div></div></div>'
            "</div>",
            unsafe_allow_html=True,
        )

        host_txt = []
        if view["host_bonus_a"]:
            host_txt.append(f"{esc(a)}: +{view['host_bonus_a']:.0f} pkt Elo")
        if view["host_bonus_b"]:
            host_txt.append(f"{esc(b)}: +{view['host_bonus_b']:.0f} pkt Elo")
        host_note = "Bonus gospodarza - " + ", ".join(host_txt) if host_txt else "Bez bonusu gospodarza"

        p1, p2 = st.columns(2)
        with p1:
            st.markdown(
                prob_bar("Czyste Elo", a, b, view["elo_prob_a"], f"{esc(a)} (niebieski) vs {esc(b)} (pomarańczowy) &middot; {host_note}"),
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                prob_bar("Model hybrydowy", a, b, view["prob_a"],
                         f"Wagi: forma {form_mult:.2f}x, styl {style_mult:.2f}x &middot; "
                         f"różnica względem Elo: {signed(view['prob_a'] - view['elo_prob_a'])} pp dla {esc(a)}"),
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="note">{commentary(view)}</div>', unsafe_allow_html=True)

        g1, g2 = st.columns([1.25, 1])
        with g1:
            st.plotly_chart(waterfall_figure(view), **STRETCH, config={"displayModeBar": False})
        with g2:
            st.plotly_chart(radar_figure(view), **STRETCH, config={"displayModeBar": False})

        section("Porównanie statystyk")
        st.markdown(stats_table_html(view), unsafe_allow_html=True)
        if note:
            st.markdown(f'<div class="small-muted" style="margin-top:8px">{note}</div>', unsafe_allow_html=True)

# =========================================================================== #
# Zakładka 3 - Monte Carlo
# =========================================================================== #
MC_STAGES = {
    "reach_round_of_16": "1/8 finału",
    "reach_quarter_final": "Ćwierćfinał",
    "reach_semi_final": "Półfinał",
    "reach_final": "Finał",
    "win_tournament": "Mistrz Świata",
}

with tab_mc:
    section("Symulacja Monte Carlo")
    st.markdown(
        '<div class="note"><b>Drabinka deterministyczna</b> pokazuje jeden, najbardziej prawdopodobny przebieg turnieju: '
        "w każdym meczu rozstrzyganym przez model awansuje faworyt, więc niespodzianki pojawiają się w niej tylko tam, "
        "gdzie wynik jest narzucony (rzeczywisty lub wybrany przez użytkownika). "
        "<b>Symulacja Monte Carlo</b> rozgrywa turniej wiele razy, a w każdym meczu losuje zwycięzcę zgodnie z "
        "prawdopodobieństwem z modelu. Rating i forma aktualizują się po każdej rundzie. Odsetek turniejów, w których "
        "drużyna dotarła do danego etapu, to jej szansa - dzięki temu widać również scenariusze mniej oczywiste, "
        "w tym ryzyko wcześniejszej porażki faworytów.</div>",
        unsafe_allow_html=True,
    )

    bcol1, bcol2 = st.columns([1, 3])
    run_clicked = bcol1.button(
        f"Przelicz symulację ({MC_SIMULATIONS:,}".replace(",", " ") + " turniejów)",
        type="primary", **STRETCH,
        help="Uruchamia symulację na żywo z bieżącym trybem i kalibracją z panelu bocznego.",
    )
    if run_clicked:
        with st.spinner("Trwa symulacja..."):
            live = run_monte_carlo(
                MC_SIMULATIONS,
                mode=mode,
                custom_overrides=overrides if mode == "sandbox" else None,
                predictor=predictor_base,
                form_multiplier=form_mult,
                style_multiplier=style_mult,
                seed=int(np.random.default_rng().integers(1, 1_000_000)),
                save=False,
            )
        st.session_state["mc_live"] = {
            "df": live,
            "label": f"{MODE_LABELS[mode]} | waga formy {form_mult:.2f}x | waga stylu {style_mult:.2f}x",
        }

    live_state = st.session_state.get("mc_live")
    if live_state:
        mc_df = live_state["df"]
        source_label = f"Wynik przeliczony na żywo: {live_state['label']}"
        if bcol2.button("Wróć do zapisanych wyników"):
            del st.session_state["mc_live"]
            st.rerun()
    else:
        mc_df = load_saved_monte_carlo(file_mtime(MONTE_CARLO_CSV))
        source_label = (
            f"Zapisane wyniki: {MC_SIMULATIONS:,}".replace(",", " ")
            + " symulacji | Prognoza przed fazą pucharową (100% AI) | kalibracja domyślna"
        )
    st.markdown(f'<div class="small-muted" style="margin:6px 0 4px">{esc(source_label)}</div>', unsafe_allow_html=True)

    mc_df = mc_df.sort_values("win_tournament", ascending=False, ignore_index=True)

    top = mc_df.head(15).iloc[::-1]
    hover = np.column_stack([top[c] for c in ("reach_quarter_final", "reach_semi_final", "reach_final")])
    fig = go.Figure(
        go.Bar(
            x=top["win_tournament"], y=top["team"], orientation="h",
            marker=dict(color=top["win_tournament"], colorscale=[[0, "#1C3A6B"], [1, COL_BLUE]], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in top["win_tournament"]], textposition="outside", cliponaxis=False,
            customdata=hover,
            hovertemplate=("<b>%{y}</b><br>Mistrz Świata: %{x:.1f}%<br>Finał: %{customdata[2]:.1f}%<br>"
                           "Półfinał: %{customdata[1]:.1f}%<br>Ćwierćfinał: %{customdata[0]:.1f}%<extra></extra>"),
        )
    )
    apply_plot_style(
        fig,
        title=dict(text="Top 15 faworytów do tytułu Mistrza Świata", font=dict(size=15, color=COL_TEXT), x=0.0),
        height=520,
        xaxis=dict(title="Szansa na tytuł (%)", range=[0, max(5.0, float(top["win_tournament"].max()) * 1.18)], ticksuffix="%"),
        yaxis=dict(title=None, automargin=True),
        bargap=0.3,
    )
    st.plotly_chart(fig, **STRETCH, config={"displayModeBar": False})

    section(
        "Szanse wszystkich 32 drużyn",
        "Odsetek symulowanych turniejów, w których drużyna osiągnęła dany etap.",
    )
    table = pd.DataFrame({"Flaga": [flag_url(t, 80) for t in mc_df["team"]], "Drużyna": mc_df["team"]})
    for col, label in MC_STAGES.items():
        table[label] = mc_df[col].astype(float)
    config = {"Flaga": st.column_config.ImageColumn(" ", width="small"),
              "Drużyna": st.column_config.TextColumn("Drużyna", width="medium")}
    for label in MC_STAGES.values():
        config[label] = st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.1f%%")
    st.dataframe(table, hide_index=True, **STRETCH, height=36 * (len(table) + 1) + 4, column_config=config)

# =========================================================================== #
# Zakładka 4 - ranking Elo
# =========================================================================== #
def set_rank_query(value: str) -> None:
    st.session_state["rank_query"] = value


with tab_rank:
    ratings = load_ratings(file_mtime(CURRENT_ELO_CSV)).copy()
    ratings["wc"] = ratings["team"].isin(WC_TEAMS)
    ratings["balance"] = ratings["goals_scored_recent"] - ratings["goals_conceded_recent"]
    ratings = ratings.sort_values("rank").reset_index(drop=True)
    n_total = len(ratings)

    section(
        "Światowy ranking Elo",
        "Rating Elo wszystkich reprezentacji FIFA, liczony na historii meczów od 1980 roku. "
        "Forma to średnia zmiana ratingu w ostatnich 8 meczach, bilans to średnia różnica bramek na mecz w tym okresie.",
    )

    f1, f2, f3 = st.columns([1.3, 2.2, 2.2])
    only_wc = f1.toggle("Tylko uczestnicy MŚ 2026 (48 drużyn)", value=False, key="rank_only_wc")
    query = f2.text_input("Szukaj drużyny", key="rank_query", placeholder="np. Poland, Polska, Brazil")
    f2.button("Pokaż Polskę", on_click=set_rank_query, args=("Poland",), key="rank_pl")
    rank_lo, rank_hi = f3.slider("Zakres miejsc w rankingu", 1, n_total, (1, n_total), key="rank_range")

    q = query.strip().lower()
    q_names = {POLISH_ALIASES[q].lower()} if q in POLISH_ALIASES else set()
    if q:
        match_mask = ratings["team"].str.lower().apply(lambda t: q in t or t in q_names)
    else:
        match_mask = pd.Series(False, index=ratings.index)

    only_matches = False
    if q:
        only_matches = f2.checkbox("Pokaż tylko dopasowane drużyny", value=False, key="rank_only_matches")

    view_mask = ratings["rank"].between(rank_lo, rank_hi)
    if only_wc:
        view_mask &= ratings["wc"]
    if only_matches:
        view_mask &= match_mask
    shown = ratings[view_mask].copy()
    shown_match = match_mask[view_mask]

    if q and match_mask.any():
        first = ratings[match_mask].iloc[0]
        extra = f" (+{int(match_mask.sum()) - 1} dopasowań)" if match_mask.sum() > 1 else ""
        st.markdown(
            f'<div class="note">{flag_img(first["team"])} <b>{esc(first["team"])}</b>{extra}: miejsce '
            f'<b>{int(first["rank"])}</b> z {n_total}, rating Elo <b>{first["current_elo"]:.0f}</b>, '
            f'forma <b>{signed(first["form_rating_change"], 2)}</b> Elo na mecz, bilans bramek '
            f'<b>{signed(first["balance"], 2)}</b> na mecz'
            f'{" &middot; uczestnik MŚ 2026" if first["wc"] else " &middot; poza MŚ 2026"}.</div>',
            unsafe_allow_html=True,
        )
    elif q:
        st.info("Brak drużyny pasującej do wyszukiwania.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Drużyny w tabeli", f"{len(shown)}")
    if len(shown):
        k2.metric("Najwyższy rating", f"{shown['current_elo'].max():.0f}", shown.loc[shown["current_elo"].idxmax(), "team"], delta_color="off")
        k3.metric("Najlepsza forma", signed(shown["form_rating_change"].max(), 2),
                  shown.loc[shown["form_rating_change"].idxmax(), "team"], delta_color="off")

    display = pd.DataFrame(
        {
            "Pozycja": shown["rank"].astype(int),
            "Flaga": [flag_url(t, 80) for t in shown["team"]],
            "Drużyna": shown["team"],
            "Rating Elo": shown["current_elo"],
            "Forma (Elo/mecz)": shown["form_rating_change"],
            "Bramki strzelone": shown["goals_scored_recent"],
            "Bramki stracone": shown["goals_conceded_recent"],
            "Bilans bramek": shown["balance"],
            "MŚ 2026": np.where(shown["wc"], "Uczestnik", ""),
        }
    ).reset_index(drop=True)
    row_is_match = shown_match.reset_index(drop=True)
    row_is_wc = shown["wc"].reset_index(drop=True)

    def color_trend(val: float) -> str:
        if val > 0.05:
            return f"color: {COL_GREEN}; font-weight: 600"
        if val < -0.05:
            return f"color: {COL_RED}; font-weight: 600"
        return f"color: {COL_MUTED}"

    def row_background(row: pd.Series) -> List[str]:
        if bool(row_is_match.iloc[row.name]):
            style = "background-color: rgba(227,179,65,.22)"
        elif bool(row_is_wc.iloc[row.name]):
            style = "background-color: rgba(76,154,255,.07)"
        else:
            style = ""
        return [style] * len(row)

    styler = display.style.apply(row_background, axis=1)
    cell_map = styler.map if hasattr(styler, "map") else styler.applymap
    styler = cell_map(color_trend, subset=["Forma (Elo/mecz)", "Bilans bramek"])
    styler = styler.format(
        {"Rating Elo": "{:.0f}", "Forma (Elo/mecz)": "{:+.2f}", "Bramki strzelone": "{:.2f}",
         "Bramki stracone": "{:.2f}", "Bilans bramek": "{:+.2f}"}
    )
    st.dataframe(
        styler,
        hide_index=True,
        **STRETCH,
        height=min(36 * (len(display) + 1) + 4, 720),
        column_config={
            "Pozycja": st.column_config.NumberColumn("Poz.", width="small"),
            "Flaga": st.column_config.ImageColumn(" ", width="small"),
            "Drużyna": st.column_config.TextColumn("Drużyna", width="medium"),
        },
    )
    st.markdown(
        '<div class="small-muted">Niebieskie tło: uczestnik MŚ 2026. Złote tło: wynik wyszukiwania. '
        "Zielony i czerwony kolor oznaczają odpowiednio wzrost i spadek formy lub dodatni i ujemny bilans bramek.</div>",
        unsafe_allow_html=True,
    )
