"""
World Cup 2026 - Knockout Bracket Predictor
============================================

Interactive Streamlit application that renders the full, official knockout
("Playoffs") bracket of the tournament - from the Round of 32 all the way
to the Final - and predicts every single match along the way.

The bracket topology and prediction logic are extracted (and lightly
adapted for a real-time UI) from this project's own simulation notebooks:

- ``notebooks/07_bracket_simulator.ipynb`` -> defines the full Round-of-32
  bracket topology (which team plays whom, and how winners feed into the
  next round: R32 -> R16 -> QF -> SF -> Final) and the Elo-based win
  probability formula used to decide every matchup.
- ``notebooks/08_deterministic_bracket.ipynb`` -> walks that exact same
  topology *deterministically* (always advancing the higher-probability
  team) to produce a single, concrete "most likely path" through the whole
  tournament, while injecting real-world results for fixtures that have
  already been played (the "Reality Check" step).

All Monte-Carlo/statistical-aggregation code (used in notebook 07 to
produce per-team advancement odds over 10,000 simulated tournaments) and
all plotting-only code were intentionally left out: this app only needs
the deterministic single-bracket walk, since it must render one concrete
tree of matches, not an aggregate report.
"""

from pathlib import Path

import math

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="World Cup 2026 | Knockout Bracket",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Global CSS - dark background, forced high-contrast WHITE text everywhere
# --------------------------------------------------------------------------- #
#
# The previous version of this app suffered from poor contrast (dark text on
# a dark background in several widgets/sidebar). This block forcefully
# overrides text color to a bright white across every Streamlit element,
# including the sidebar and dropdown/select widgets, while keeping the dark
# themed background untouched.

GLOBAL_CSS = """
<style>
/* ---- App background (kept dark) ---- */
.stApp {
    background: radial-gradient(circle at top left, #123524 0%, #081812 55%, #04090a 100%);
}

/* ---- Force bright white text EVERYWHERE ---- */
html, body, .stApp, [class*="css"] {
    color: #FFFFFF !important;
}

h1, h2, h3, h4, h5, h6,
p, span, div, label, li, small, strong, em, a,
[data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"],
[data-testid="stText"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p {
    color: #FFFFFF !important;
}

/* ---- Sidebar: dark background + forced white text on every child ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2a1f 0%, #061510 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.15);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* ---- Select / dropdown widgets: dark surface so white text stays legible ---- */
[data-baseweb="select"] > div {
    background-color: #12331f !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
    color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background-color: #12331f !important;
}
[role="option"] {
    color: #FFFFFF !important;
    background-color: #12331f !important;
}
[role="option"]:hover,
[aria-selected="true"] {
    background-color: #1c7a4d !important;
    color: #FFFFFF !important;
}
[data-baseweb="select"] svg {
    fill: #FFFFFF !important;
}

/* ---- Inline code snippets (e.g. `file.csv`) - keep them legible ---- */
code, .stMarkdown code, [data-testid="stCaptionContainer"] code {
    color: #ffce54 !important;
    background-color: rgba(255, 255, 255, 0.14) !important;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    padding: 1px 5px;
}

/* ---- Misc widget chrome ---- */
hr, [data-testid="stDivider"] {
    border-color: rgba(255, 255, 255, 0.25) !important;
}
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 10px;
}

footer {visibility: hidden;}

/* --------------------------------------------------------------------- */
/* Bracket-specific styling                                              */
/* --------------------------------------------------------------------- */

.app-hero {
    padding: 1.5rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #0d3b26 0%, #145c3a 45%, #1c7a4d 100%);
    border: 1px solid rgba(255, 255, 255, 0.25);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.2rem;
    text-align: center;
}
.app-hero h1 {
    margin: 0;
    font-size: 2rem;
    letter-spacing: 0.5px;
}
.app-hero p {
    margin: 0.35rem 0 0 0;
    font-size: 1rem;
    opacity: 0.9;
}

.round-title {
    text-align: center;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 0.5rem 0.4rem;
    margin-bottom: 0.6rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.bracket-col {
    display: flex;
    flex-direction: column;
}

.match-slot {
    flex: 1 1 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 3px 2px;
}

.match-card {
    width: 100%;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.78rem;
}
.match-card.is-real {
    border-color: rgba(255, 206, 84, 0.65);
    box-shadow: 0 0 10px rgba(255, 206, 84, 0.15);
}

.team-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    padding: 5px 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}
.team-row:last-child {
    border-bottom: none;
}
.team-row.winner {
    background: rgba(28, 122, 77, 0.55);
    font-weight: 800;
}
.team-row.loser {
    opacity: 0.5;
}
.team-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.team-prob {
    font-weight: 700;
    font-size: 0.72rem;
    flex-shrink: 0;
}

.real-tag {
    display: block;
    text-align: center;
    font-size: 0.62rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    font-weight: 700;
    color: #1a1608 !important;
    background: #ffce54;
    padding: 1px 0;
}

.champion-slot {
    flex: 1 1 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.champion-card {
    text-align: center;
    padding: 1.4rem 1rem;
    border-radius: 16px;
    background: linear-gradient(160deg, #7a5c00 0%, #ad8800 45%, #ffce54 100%);
    border: 2px solid #ffe9a8;
    box-shadow: 0 0 25px rgba(255, 206, 84, 0.45);
    width: 100%;
}
.champion-card .trophy {
    font-size: 2.4rem;
    display: block;
    margin-bottom: 0.3rem;
}
.champion-card .champion-name {
    font-size: 1.25rem;
    font-weight: 900;
    color: #1a1608 !important;
    letter-spacing: 0.5px;
}
.champion-card .champion-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #3a2e00 !important;
    font-weight: 700;
}

.legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 10px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 0.78rem;
    margin-right: 10px;
}

/* ---- "Ranking Elo" legend chip for World Cup 2026 teams ---- */
.legend-chip.wc2026 {
    background: #332b00;
    border: 1px solid #ffce54;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Paths & data
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
ELO_PATH = BASE_DIR / "data" / "processed" / "current_elo_ratings.csv"
GROUP_STAGE_STATS_PATH = BASE_DIR / "data" / "processed" / "fbref_group_stats.csv"


@st.cache_data(show_spinner=False)
def load_elo_ratings() -> dict:
    df_elo = pd.read_csv(ELO_PATH)
    return dict(zip(df_elo["team"], df_elo["current_elo"]))


# `fbref_group_stats.csv` (scraped in `notebooks/03_fbref_scraping.ipynb`)
# has exactly one row per team that played in the World Cup 2026 group
# stage - i.e. the full list of the 48 participating national teams. FBref
# spells a handful of names differently than the historical-results
# dataset that powers the Elo ratings, so they're normalized to the
# spelling used everywhere else in this app (and in
# `current_elo_ratings.csv`) before being used for matching.
FBREF_TO_ELO_NAME = {
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
}


@st.cache_data(show_spinner=False)
def load_world_cup_2026_teams() -> set:
    """The 48 national teams competing at the World Cup 2026."""
    df_groups = pd.read_csv(GROUP_STAGE_STATS_PATH)
    normalized = df_groups["team"].replace(FBREF_TO_ELO_NAME)
    return set(normalized)


WORLD_CUP_2026_TEAMS = load_world_cup_2026_teams()


@st.cache_data(show_spinner=False)
def load_elo_ratings_table() -> tuple:
    """Full Elo leaderboard (`data/processed/current_elo_ratings.csv`),
    sorted from strongest to weakest team - used to populate the
    "Ranking Elo" tab. This file is produced by
    `notebooks/02_elo_engine.ipynb`, which now only considers matches
    played from 2016 onward, so the ratings reflect the last 10 years
    of team form.

    Teams competing at the World Cup 2026 get a trailing 🏆 marker on
    their name so they stand out immediately in a list of ~300 teams.
    Returns (dataframe, is_world_cup_2026_mask) - the boolean mask is
    kept separate from the dataframe so it can drive row highlighting
    without becoming a visible column.
    """
    df_elo = pd.read_csv(ELO_PATH)
    df_elo = df_elo.sort_values(by="current_elo", ascending=False).reset_index(drop=True)
    df_elo.insert(0, "Pozycja", df_elo.index + 1)

    is_wc_2026 = df_elo["team"].isin(WORLD_CUP_2026_TEAMS)
    df_elo.loc[is_wc_2026, "team"] = df_elo.loc[is_wc_2026, "team"] + " 🏆"

    df_elo = df_elo.rename(columns={"team": "Drużyna", "current_elo": "Rating ELO"})
    return df_elo, is_wc_2026


ELO_RATINGS = load_elo_ratings()
ELO_RATINGS_TABLE, ELO_RATINGS_IS_WC_2026 = load_elo_ratings_table()

# --------------------------------------------------------------------------- #
# Bracket topology
# (notebooks/07_bracket_simulator.ipynb & notebooks/08_deterministic_bracket.ipynb)
# --------------------------------------------------------------------------- #
#
# Left half (8 matches) followed by right half (8 matches) of the Round of
# 32. Every later round is simply "pair up consecutive winners", which
# exactly reproduces the L1-L8 / R16_L.. / QF_L.. / SF_L (and R.. mirror)
# progression coded explicitly in the notebooks.

ROUND_OF_32_FIXTURES = [
    ("Germany", "Paraguay"),
    ("France", "Sweden"),
    ("South Africa", "Canada"),
    ("Netherlands", "Morocco"),
    ("Portugal", "Croatia"),
    ("Spain", "Austria"),
    ("United States", "Bosnia and Herzegovina"),
    ("Belgium", "Senegal"),
    ("Brazil", "Japan"),
    ("Ivory Coast", "Norway"),
    ("Mexico", "Ecuador"),
    ("England", "DR Congo"),
    ("Argentina", "Cape Verde"),
    ("Australia", "Egypt"),
    ("Switzerland", "Algeria"),
    ("Colombia", "Ghana"),
]

# "Reality Check": fixtures that have already been played in real life, with
# their actual outcome injected instead of relying on the Elo model.
# (notebooks/08_deterministic_bracket.ipynb)
#
# Stan na dzień dzisiejszy: cała runda 1/16 finału (Round of 32) ORAZ 6 z 8
# meczów 1/8 finału (Round of 16) zostały już rozegrane w rzeczywistości.
# Pozostałe mecze (2 mecze 1/8 finału, ćwierćfinały, półfinały, finał) wciąż
# czekają na rozegranie, więc nadal są prognozowane przez model Elo.
#
# WAŻNE: to jest twardy "override" - `predict_match()` (poniżej) ZAWSZE
# zwraca zwycięzcę wpisanego tutaj, niezależnie od tego, co mówi model Elo,
# nawet jeśli dana drużyna miała mniejsze procentowe szanse na wygraną.
#
# Klucze są `frozenset({team_a, team_b})` (a nie tuple) - dopasowanie meczu
# jest więc z definicji NIEZALEŻNE od kolejności drużyn: nie ma znaczenia,
# czy w danej rundzie drabinki dana drużyna wypadnie jako "team_a" czy
# "team_b", bo frozenset({"A", "B"}) == frozenset({"B", "A"}).
KNOWN_RESULTS = {
    frozenset({'Colombia', 'Ghana'}): 'Colombia',
    frozenset({'Argentina', 'Cape Verde'}): 'Argentina',
    frozenset({'Egypt', 'Australia'}): 'Egypt',
    frozenset({'Switzerland', 'Algeria'}): 'Switzerland',
    frozenset({'Portugal', 'Croatia'}): 'Portugal',
    frozenset({'Spain', 'Austria'}): 'Spain',
    frozenset({'United States', 'Bosnia and Herzegovina'}): 'United States',
    frozenset({'Belgium', 'Senegal'}): 'Belgium',
    frozenset({'England', 'DR Congo'}): 'England',
    frozenset({'Mexico', 'Ecuador'}): 'Mexico',
    frozenset({'France', 'Sweden'}): 'France',
    frozenset({'Norway', 'Ivory Coast'}): 'Norway',
    frozenset({'Morocco', 'Netherlands'}): 'Morocco',
    frozenset({'Paraguay', 'Germany'}): 'Paraguay',
    frozenset({'Brazil', 'Japan'}): 'Brazil',
    frozenset({'Canada', 'South Africa'}): 'Canada',
    frozenset({'Belgium', 'United States'}): 'Belgium',
    frozenset({'Spain', 'Portugal'}): 'Spain',
    frozenset({'England', 'Mexico'}): 'England',
    frozenset({'Norway', 'Brazil'}): 'Norway',
    frozenset({'France', 'Paraguay'}): 'France',
    frozenset({'Morocco', 'Canada'}): 'Morocco'
}

ROUND_LABELS = [
    "1/16 Finału",
    "1/8 Finału",
    "Ćwierćfinał",
    "Półfinał",
    "Finał",
]


def elo_win_probability(team_a: str, team_b: str) -> float:
    """Classic Elo expected-score formula, same as used across every
    notebook in this project (02, 07, 08)."""
    elo_a = ELO_RATINGS.get(team_a, 1500)
    elo_b = ELO_RATINGS.get(team_b, 1500)
    return 1.0 / (1.0 + math.pow(10, (elo_b - elo_a) / 400.0))


def predict_match(team_a: str, team_b: str) -> dict:
    """Deterministic single-match prediction, mirroring
    notebooks/08_deterministic_bracket.ipynb.

    HARD OVERRIDE RULE: if the fixture is present in ``KNOWN_RESULTS``, that
    real-world result is *unconditionally* the winner - the Elo model's
    opinion is never consulted to decide who advances, even when the real
    winner had a lower/losing Elo probability (an "upset"). Elo is only
    ever used to pick a winner when the match has genuinely not been played
    yet.

    Matching against ``KNOWN_RESULTS`` is entirely order-independent: the
    lookup key is ``frozenset({team_a, team_b})``, so it doesn't matter
    whether a given team ends up as "team_a" or "team_b" in a particular
    round of the bracket walk.

    The Elo win probability is always computed and returned regardless of
    which branch decided the winner: the UI still shows what odds the model
    gave *before* the match was played, alongside the real-result marker.
    """
    prob_a = elo_win_probability(team_a, team_b)
    prob_b = 1.0 - prob_a

    winner = KNOWN_RESULTS.get(frozenset({team_a, team_b}))

    # Defensive fallback (should never trigger with well-formed data): if a
    # KNOWN_RESULTS entry somehow names a winner that isn't either team in
    # this fixture, don't crash the whole app - just ignore that (broken)
    # entry and let the Elo model decide this particular match instead.
    if winner is not None and winner not in (team_a, team_b):
        winner = None

    if winner is not None:
        return {"team_a": team_a, "team_b": team_b, "winner": winner,
                "prob_a": prob_a * 100, "prob_b": prob_b * 100, "is_real": True}

    winner = team_a if prob_a > 0.5 else team_b
    return {"team_a": team_a, "team_b": team_b, "winner": winner,
             "prob_a": prob_a * 100, "prob_b": prob_b * 100, "is_real": False}


def find_unmatched_known_results(rounds: list) -> set:
    """Sanity check for the hard-override rule above (non-fatal): returns
    every ``KNOWN_RESULTS`` entry that was never matched against an actual
    fixture while walking the *entire* bracket (every round returned by
    ``simulate_full_bracket`` - Round of 32, Round of 16, quarter-finals,
    semi-finals and the final, not just the initial ``ROUND_OF_32_FIXTURES``
    list). A non-empty result usually means a team name in ``KNOWN_RESULTS``
    doesn't exactly match the name used elsewhere in the app - this is only
    used to show an informational note in the UI, it never blocks the app
    from starting."""
    matched_keys = set()
    for round_matches in rounds:
        for match in round_matches:
            if not match["is_real"]:
                continue
            matched_keys.add(frozenset({match["team_a"], match["team_b"]}))
    return set(KNOWN_RESULTS) - matched_keys


# NOTE: deliberately NOT decorated with @st.cache_data. This function's
# only argument (`fixtures`) never changes between reruns, but its actual
# output depends on the *global* `KNOWN_RESULTS` dict (read indirectly via
# `predict_match`). Streamlit's cache key is derived from the function's own
# source code + its explicit arguments - it has no way to know that a global
# it reads through a nested call has changed. Caching this here previously
# meant that editing `KNOWN_RESULTS` (e.g. adding/fixing real-world results)
# could silently keep serving a stale bracket from before the edit, for as
# long as the Streamlit process stayed warm. The whole walk is only ~31
# matches, so recomputing it on every rerun is effectively free - not worth
# the staleness risk.
def simulate_full_bracket(fixtures: tuple) -> list:
    """Walks the entire Round-of-32 -> Final topology one round at a time,
    exactly like the `simulate_side` loop in
    notebooks/08_deterministic_bracket.ipynb, but flattened across the whole
    32-team field instead of splitting into a left/right half. Pairing
    consecutive winners round after round reproduces that exact bracket
    structure. Returns a list of rounds, each a list of match-result dicts.
    """
    rounds = []
    current_round_teams = list(fixtures)

    while True:
        match_results = [predict_match(a, b) for a, b in current_round_teams]
        rounds.append(match_results)
        winners = [m["winner"] for m in match_results]
        if len(winners) == 1:
            break
        current_round_teams = list(zip(winners[0::2], winners[1::2]))

    return rounds


bracket_rounds = simulate_full_bracket(tuple(ROUND_OF_32_FIXTURES))
champion = bracket_rounds[-1][0]["winner"]

# Non-fatal sanity check: never blocks the app from starting. If it finds
# something, it's surfaced as a small info note in the sidebar (see below)
# instead of crashing - predict_match() overriding known winners correctly
# is what matters, not this diagnostic.
UNMATCHED_KNOWN_RESULTS = find_unmatched_known_results(bracket_rounds)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("## 🏆 Drabinka Playoffs")
    st.caption(
        "Pełna, deterministyczna drabinka fazy pucharowej Mistrzostw Świata 2026 "
        "- od 1/16 finału aż do Finału."
    )
    st.divider()
    st.markdown("### 🧮 Jak liczony jest wynik?")
    st.caption(
        "• Mecze **już rozegrane** ⭐ pobierają rzeczywisty wynik "
        "(`known_winners`, zgodnie z `08_deterministic_bracket.ipynb`) - to "
        "**twardy override**: liczy się wyłącznie rzeczywisty wynik, nawet "
        "jeśli dana drużyna miała niższe szanse wg Elo. Pod nazwami drużyn "
        "wciąż widać procentowe szanse, jakie dawał model **przed** "
        "rozegraniem meczu.\n\n"
        "• Mecze **przyszłe** rozstrzyga model ratingu **Elo** "
        "(`02_elo_engine.ipynb`): wygrywa drużyna z szansą > 50%."
    )
    st.divider()
    st.markdown("### 🔑 Legenda")
    st.markdown(
        '<span class="legend-chip">⭐ Wynik rzeczywisty</span>'
        '<span class="legend-chip">🔮 Prognoza Elo</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "📑 **Zakładki:** *Drabinka Turniejowa* (prognoza fazy pucharowej) "
        "oraz *Ranking Elo* (pełna tabela ratingów wszystkich drużyn)."
    )
    st.divider()
    st.caption("Dane: `data/processed/current_elo_ratings.csv`")
    st.caption("Logika: `notebooks/07_bracket_simulator.ipynb`, `notebooks/08_deterministic_bracket.ipynb`")

    if UNMATCHED_KNOWN_RESULTS:
        # Informational only - never blocks the app. See
        # find_unmatched_known_results() for details.
        _unmatched_teams = ", ".join(
            " vs ".join(pair) for pair in sorted(sorted(p) for p in UNMATCHED_KNOWN_RESULTS)
        )
        st.info(
            f"ℹ️ Uwaga deweloperska: {len(UNMATCHED_KNOWN_RESULTS)} wpis(y) w "
            f"`KNOWN_RESULTS` nie trafiły na żaden mecz w drabince ({_unmatched_teams}). "
            "Nie wpływa to na działanie aplikacji - prognoza jest generowana normalnie."
        )

# --------------------------------------------------------------------------- #
# Main view - hero header
# --------------------------------------------------------------------------- #

tab1, tab2 = st.tabs(["Drabinka Turniejowa", "Ranking ELO"])

with tab1:
    st.markdown(
        '<div class="app-hero">'
        '<h1>🏆 Mistrzostwa Świata 2026 &mdash; Drabinka Fazy Pucharowej</h1>'
        '<p>Deterministyczna prognoza całego turnieju: 1/16 finału → 1/8 finału → Ćwierćfinał → Półfinał → Finał</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    legend_col_a, legend_col_b, legend_col_c = st.columns(3)
    legend_col_a.metric("Drużyny w drabince", "32")
    legend_col_b.metric("Mecze do rozegrania", str(sum(len(r) for r in bracket_rounds)))
    legend_col_c.metric("Przewidywany Mistrz 🏆", champion)

    st.divider()

    # --------------------------------------------------------------------------- #
    # Bracket rendering
    # --------------------------------------------------------------------------- #

    TOTAL_SLOTS = len(bracket_rounds[0])  # 16 Round-of-32 matches -> tallest column
    BRACKET_HEIGHT_PX = TOTAL_SLOTS * 108


    # NOTE: every HTML string built below is assembled with ZERO leading
    # whitespace on each concatenated fragment and joined without newlines.
    # Streamlit's markdown renderer treats lines indented by 4+ spaces (or
    # certain blank-line/indentation combinations) as Markdown *code blocks*
    # even with unsafe_allow_html=True, which silently breaks HTML rendering.
    # Keeping everything on tightly joined, non-indented single-line fragments
    # avoids that pitfall entirely.


    def render_match_card(match: dict) -> str:
        team_a, team_b, winner = match["team_a"], match["team_b"], match["winner"]
        is_real = match["is_real"]

        # Mecze rozegrane pokazują GWIAZDKĘ *i* procentowe szanse wyliczone
        # przez model Elo - tak użytkownik widzi, jakiego wyniku oczekiwał
        # model, zanim mecz się faktycznie odbył.
        if is_real:
            prob_a_label = f"⭐ {match['prob_a']:.0f}%"
            prob_b_label = f"⭐ {match['prob_b']:.0f}%"
        else:
            prob_a_label = f"{match['prob_a']:.0f}%"
            prob_b_label = f"{match['prob_b']:.0f}%"

        row_a_cls = "winner" if winner == team_a else "loser"
        row_b_cls = "winner" if winner == team_b else "loser"
        card_cls = "match-card is-real" if is_real else "match-card"

        real_tag = '<span class="real-tag">Wynik rzeczywisty</span>' if is_real else ""

        return (
            f'<div class="{card_cls}">'
            f"{real_tag}"
            f'<div class="team-row {row_a_cls}">'
            f'<span class="team-name">{team_a}</span>'
            f'<span class="team-prob">{prob_a_label}</span>'
            f"</div>"
            f'<div class="team-row {row_b_cls}">'
            f'<span class="team-name">{team_b}</span>'
            f'<span class="team-prob">{prob_b_label}</span>'
            f"</div>"
            f"</div>"
        )


    def render_round_column(round_matches: list, label: str) -> str:
        slots_html = "".join(
            f'<div class="match-slot">{render_match_card(m)}</div>' for m in round_matches
        )
        return (
            f'<div class="round-title">{label}</div>'
            f'<div class="bracket-col" style="height:{BRACKET_HEIGHT_PX}px;">'
            f"{slots_html}"
            f"</div>"
        )


    def render_champion_column(champion_team: str) -> str:
        return (
            f'<div class="round-title">Mistrz Świata</div>'
            f'<div class="bracket-col" style="height:{BRACKET_HEIGHT_PX}px;">'
            f'<div class="champion-slot">'
            f'<div class="champion-card">'
            f'<span class="trophy">🏆</span>'
            f'<span class="champion-label">Mistrz Świata 2026</span><br/>'
            f'<span class="champion-name">{champion_team}</span>'
            f"</div>"
            f"</div>"
            f"</div>"
        )


    bracket_columns = st.columns(len(bracket_rounds) + 1)

    for col, round_matches, label in zip(bracket_columns[:-1], bracket_rounds, ROUND_LABELS):
        with col:
            st.markdown(render_round_column(round_matches, label), unsafe_allow_html=True)

    with bracket_columns[-1]:
        st.markdown(render_champion_column(champion), unsafe_allow_html=True)

    st.divider()
    st.caption(
        "⚠️ Drabinka jest generowana automatycznie na bazie ratingu Elo oraz znanych "
        "wyników rzeczywistych mocno wczesnych meczów. Prognozy służą celom "
        "demonstracyjnym/portfolio i nie stanowią porady bukmacherskiej."
    )

    with st.expander("🔧 Szczegóły techniczne symulacji"):
        st.markdown(
            """
            - **Topologia drabinki:** 16 par 1/16 finału (32 drużyny), redukowane
              konsekwentnie do 1/8 finału, ćwierćfinału, półfinału i finału
              &mdash; identyczna struktura jak w `notebooks/07_bracket_simulator.ipynb`
              i `notebooks/08_deterministic_bracket.ipynb`.
            - **Reguła awansu:** dla każdego meczu wygrywa drużyna z wyższym
              prawdopodobieństwem wygranej wg wzoru Elo:
              `P(A) = 1 / (1 + 10^((Elo_B - Elo_A) / 400))`.
            - **Wstrzykiwanie realnych wyników (twardy override):** dla meczów już
              rozegranych w rzeczywistości, model nie zgaduje &mdash; wynik jest
              wzięty bezpośrednio z danych (`KNOWN_RESULTS`, odpowiednik
              `known_winners` z notatnika 08) i **zawsze** wygrywa, nawet gdy
              Elo wskazywało inną drużynę jako bardziej prawdopodobnego
              zwycięzcę. Dopasowanie par drużyn jest niezależne od kolejności
              (`frozenset({team_a, team_b})` jako klucz słownika).
            - **Diagnostyka (nieblokująca):** `find_unmatched_known_results()`
              przeszukuje całą wygenerowaną drabinkę (1/16, 1/8, ćwierćfinały,
              półfinały, finał) i wyłącznie informacyjnie zgłasza w panelu
              bocznym wpisy `KNOWN_RESULTS`, które nie trafiły na żaden mecz
              &mdash; nigdy nie blokuje to działania aplikacji.
            - **Różnica względem notatnika 08:** logika przechodzenia przez rundy
              jest tu spłaszczona (lista kolejnych par, redukowana runda po
              rundzie) zamiast jawnego podziału na stronę LEFT/RIGHT &mdash; daje
              to identyczne wyniki, ale prościej skaluje się do renderowania w UI.
            """
        )

# --------------------------------------------------------------------------- #
# "Ranking ELO" tab - full leaderboard from current_elo_ratings.csv
# --------------------------------------------------------------------------- #

with tab2:
    st.markdown(
        '<div class="app-hero">'
        '<h1>📊 Ranking Elo &mdash; Wszystkie Drużyny</h1>'
        '<p>Aktualny rating Elo każdej drużyny, wyliczony na podstawie meczów '
        'z ostatnich 10 lat (od 2016 roku) - <code>02_elo_engine.ipynb</code></p>'
        '</div>',
        unsafe_allow_html=True,
    )

    rank_col_a, rank_col_b, rank_col_c, rank_col_d = st.columns(4)
    rank_col_a.metric("Drużyny w rankingu", str(len(ELO_RATINGS_TABLE)))
    rank_col_b.metric("Reprezentacje MŚ 2026 🏆", str(int(ELO_RATINGS_IS_WC_2026.sum())))
    rank_col_c.metric("Liderzy rankingu 🥇", ELO_RATINGS_TABLE.iloc[0]["Drużyna"])
    rank_col_d.metric(
        "Najwyższy rating",
        f"{ELO_RATINGS_TABLE.iloc[0]['Rating ELO']:.0f}",
    )

    st.markdown(
        '<span class="legend-chip wc2026">🏆 Uczestnik Mistrzostw Świata 2026</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    show_only_wc_2026 = st.toggle(
        "Pokaż tylko uczestników MŚ 2026",
        value=True,
        help="Włączone: tabela zawiera wyłącznie 48 reprezentacji biorących "
             "udział w Mistrzostwach Świata 2026. Wyłączone: pełna lista "
             "wszystkich drużyn w rankingu Elo.",
    )

    if show_only_wc_2026:
        display_ratings_table = ELO_RATINGS_TABLE.loc[ELO_RATINGS_IS_WC_2026].reset_index(drop=True)
        display_is_wc_2026 = pd.Series(True, index=display_ratings_table.index)
    else:
        display_ratings_table = ELO_RATINGS_TABLE
        display_is_wc_2026 = ELO_RATINGS_IS_WC_2026

    def highlight_world_cup_2026_rows(row: pd.Series) -> list:
        """Row-wise style function for `Styler.apply`: gives every World
        Cup 2026 team a subtle dark-gold background so the 48 relevant
        teams jump out from the ~300-team leaderboard, while staying
        legible against the app's dark theme."""
        is_participant = display_is_wc_2026.iloc[row.name]
        style = "background-color: #332b00;" if is_participant else ""
        return [style] * len(row)

    styled_ratings_table = display_ratings_table.style.apply(highlight_world_cup_2026_rows, axis=1)

    # Centered, fixed-width table: a 3-column layout with a wide middle
    # column keeps the leaderboard from stretching edge-to-edge on wide
    # screens while still looking centered.
    table_col_left, table_col_center, table_col_right = st.columns([1, 3, 1])

    with table_col_center:
        st.dataframe(
            styled_ratings_table,
            hide_index=True,
            use_container_width=True,
            height=min(52 * (len(display_ratings_table) + 1), 1600),
            column_config={
                "Pozycja": st.column_config.NumberColumn("Pozycja", width="small"),
                "Drużyna": st.column_config.TextColumn("Drużyna", width="medium"),
                "Rating ELO": st.column_config.NumberColumn(
                    "Rating ELO",
                    format="%.0f",
                    width="medium",
                ),
            },
        )

    st.divider()
    st.caption(
        "Dane: `data/processed/current_elo_ratings.csv` (regenerowane przez "
        "`notebooks/02_elo_engine.ipynb`, wyłącznie na podstawie meczów "
        "rozegranych od 2016 roku). Reprezentacje Mistrzostw Świata 2026: "
        "`data/processed/fbref_group_stats.csv`."
    )
