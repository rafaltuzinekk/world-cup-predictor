"""
World Cup 2026 Match Predictor
==============================

Interactive Streamlit application that lets a user pick two national teams
and see the model-generated probability of a Home Win / Draw / Away Win.

The predictive logic here is isolated (and lightly extended) from the
project's own Data Science pipeline:

- ``notebooks/02_elo_engine.ipynb``      -> builds ``current_elo_ratings.csv``
- ``notebooks/04_feature_engineering.ipynb`` -> blends Elo with FBref form
  stats (possession / goals) into team "differentials".
- ``notebooks/05_model_training.ipynb``  -> trains a ``RandomForestClassifier``
  on the historical Elo difference and applies a "Form Index" modifier on
  top of the raw model output.

The original notebook (05) only trains a *binary* Home-Win vs. Not-Home-Win
classifier, because it is only ever used for knockout fixtures where a draw
is not a valid outcome. Since this app needs a genuine three-way probability
(Win A / Draw / Win B) for *any* pair of teams, the same Random Forest recipe
(same features, same hyperparameters) is retrained here on a 3-class target
(Away Win / Draw / Home Win) using the full historical dataset. The
"Form Index" blending step from the notebook is preserved unchanged.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------------------- #
# Page configuration & styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="World Cup 2026 | AI Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at top left, #0f3d2e 0%, #06110c 55%, #050a08 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b2a1f 0%, #071812 100%);
    border-right: 1px solid rgba(0, 230, 118, 0.25);
}

.app-hero {
    padding: 1.6rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #0d3b26 0%, #145c3a 45%, #1c7a4d 100%);
    border: 1px solid rgba(0, 230, 118, 0.35);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.4rem;
}
.app-hero h1 {
    margin: 0;
    font-size: 2.1rem;
    color: #eafff2;
    letter-spacing: 0.5px;
}
.app-hero p {
    margin: 0.35rem 0 0 0;
    color: #c8f5da;
    font-size: 1.02rem;
}

.matchup-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(0, 230, 118, 0.22);
    border-radius: 16px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.matchup-card h2 {
    color: #f5fff9;
    margin-bottom: 0.15rem;
    font-size: 1.5rem;
}
.matchup-card .subtitle {
    color: #9fd8b6;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.vs-badge {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}
.vs-badge span {
    font-weight: 800;
    font-size: 1.6rem;
    color: #ffce54;
    border: 2px solid #ffce54;
    border-radius: 50%;
    width: 62px;
    height: 62px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 206, 84, 0.08);
}

.verdict-banner {
    text-align: center;
    padding: 0.9rem 1rem;
    border-radius: 14px;
    background: rgba(0, 230, 118, 0.08);
    border: 1px solid rgba(0, 230, 118, 0.3);
    color: #eafff2;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 1rem 0 1.4rem 0;
}

.section-title {
    color: #eafff2;
    font-weight: 700;
    font-size: 1.15rem;
    margin: 1.6rem 0 0.6rem 0;
    border-left: 4px solid #1c7a4d;
    padding-left: 0.6rem;
}

footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

HISTORICAL_ELO_PATH = DATA_DIR / "historical_with_elo.csv"
CURRENT_ELO_PATH = DATA_DIR / "current_elo_ratings.csv"
FORM_STATS_PATH = DATA_DIR / "fbref_group_stats.csv"


# --------------------------------------------------------------------------- #
# Data loading & team profile construction
# (mirrors notebooks/04_feature_engineering.ipynb)
# --------------------------------------------------------------------------- #


def standardize_team_name(name: str) -> str:
    """Reconciles the different team-name spellings used by the Elo engine
    (historical results dataset) and the FBref scraper, so both sources can
    be merged into a single team profile. Extends the original notebook's
    mapping with a couple of extra aliases (Czechia, Turkiye) that were
    previously falling through and being dropped from the tournament list.
    """
    name = str(name).strip()
    if "Bosnia" in name:
        return "Bosnia and Herzegovina"
    if "USA" in name or name == "US":
        return "United States"
    if "Korea" in name and "South" not in name and "North" not in name and "United" not in name:
        return "South Korea"
    if "Iran" in name:
        return "Iran"
    if "Cote" in name or "Côte" in name or "Ivory" in name:
        return "Ivory Coast"
    if "Cabo Verde" in name:
        return "Cape Verde"
    if "Congo" in name and "DR" in name:
        return "DR Congo"
    if "Czech" in name:
        return "Czech Republic"
    if "Turk" in name or "Türk" in name:
        return "Turkey"
    return name


@st.cache_data(show_spinner=False)
def load_team_profiles() -> pd.DataFrame:
    """Builds a clean per-team profile (Elo rating + tournament form stats)
    for every 2026 World Cup team, ready to be turned into model features.
    """
    elo_df = pd.read_csv(CURRENT_ELO_PATH)
    stats_df = pd.read_csv(FORM_STATS_PATH)

    elo_df["team"] = elo_df["team"].apply(standardize_team_name)
    stats_df["team"] = stats_df["team"].apply(standardize_team_name)

    # Some historical team names collapse onto the same standardized name
    # (e.g. a defunct/diaspora side sharing a substring with "Korea"). Keep
    # the highest-rated (most representative) Elo entry for each name.
    elo_df = elo_df.sort_values("current_elo", ascending=False)

    profiles = pd.merge(
        elo_df, stats_df[["team", "Poss", "Gls"]], on="team", how="inner"
    )
    profiles = profiles.drop_duplicates(subset="team", keep="first")
    profiles = profiles.sort_values("current_elo", ascending=False).reset_index(drop=True)
    return profiles


@st.cache_resource(show_spinner=False)
def train_outcome_model() -> RandomForestClassifier:
    """Trains the Random Forest classifier following the exact recipe of
    notebooks/05_model_training.ipynb (same feature -- Elo difference --
    and same hyperparameters), but on a 3-class target (Away Win / Draw /
    Home Win) so the app can report a genuine draw probability.
    """
    history = pd.read_csv(HISTORICAL_ELO_PATH)
    history["elo_diff"] = history["home_elo_before"] - history["away_elo_before"]

    conditions = [
        history["home_score"] > history["away_score"],
        history["home_score"] == history["away_score"],
    ]
    # 2 = Home Win, 1 = Draw, 0 = Away Win
    history["target"] = np.select(conditions, [2, 1], default=0)

    history = history.dropna(subset=["elo_diff"])

    X_train = history[["elo_diff"]]
    y_train = history["target"]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict_match(
    model: RandomForestClassifier,
    team_a: pd.Series,
    team_b: pd.Series,
) -> tuple[float, float, float]:
    """Returns (P(Team A wins), P(Draw), P(Team B wins)).

    Reproduces the "Form Index Blending" step from
    notebooks/05_model_training.ipynb: the raw historical-Elo probability is
    nudged by a modifier derived from the possession/goal differential of
    the two teams' current tournament form, then re-normalized so all three
    outcomes still sum to 100%.
    """
    elo_diff = team_a["current_elo"] - team_b["current_elo"]
    poss_diff = team_a["Poss"] - team_b["Poss"]
    gls_diff = team_a["Gls"] - team_b["Gls"]

    features = pd.DataFrame({"elo_diff": [elo_diff]})
    raw_probs = model.predict_proba(features)[0]
    class_to_prob = dict(zip(model.classes_, raw_probs))

    p_away_win = class_to_prob.get(0, 0.0)
    p_draw = class_to_prob.get(1, 0.0)
    p_home_win = class_to_prob.get(2, 0.0)

    form_modifier = (poss_diff * 0.002) + (gls_diff * 0.02)
    p_home_win_adj = np.clip(p_home_win + form_modifier, 0.01, 0.97)
    p_away_win_adj = np.clip(p_away_win - form_modifier, 0.01, 0.97)
    p_draw_adj = p_draw

    total = p_home_win_adj + p_draw_adj + p_away_win_adj
    return p_home_win_adj / total, p_draw_adj / total, p_away_win_adj / total


# --------------------------------------------------------------------------- #
# Load data & model
# --------------------------------------------------------------------------- #

profiles = load_team_profiles()
model = train_outcome_model()
team_names = profiles["team"].tolist()

# --------------------------------------------------------------------------- #
# Sidebar - team selection
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("## ⚽ Ustawienia Meczu")
    st.caption("Wybierz dwie drużyny, aby wygenerować prognozę AI.")

    st.markdown("### 🔵 Drużyna A")
    team_a_name = st.selectbox(
        "Wybierz Drużynę A",
        options=team_names,
        index=team_names.index("Argentina") if "Argentina" in team_names else 0,
        key="team_a",
        label_visibility="collapsed",
    )

    st.markdown("### 🔴 Drużyna B")
    default_b_idx = team_names.index("France") if "France" in team_names else min(1, len(team_names) - 1)
    team_b_name = st.selectbox(
        "Wybierz Drużynę B",
        options=team_names,
        index=default_b_idx,
        key="team_b",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### ℹ️ O modelu")
    st.caption(
        "Model bazuje na **Random Forest** wytrenowanym na "
        "150+ latach historii meczów międzynarodowych (rating Elo), "
        "wzbogaconym o bieżącą formę drużyn (posiadanie piłki, gole) "
        "z turnieju FBref."
    )
    st.caption("Dane: `data/processed/` • Logika modelu: `notebooks/05_model_training.ipynb`")

# --------------------------------------------------------------------------- #
# Main view
# --------------------------------------------------------------------------- #

st.markdown(
    """
    <div class="app-hero">
        <h1>🏆 AI Predyktor Meczów &mdash; Mistrzostwa Świata 2026</h1>
        <p>Statystyczna prognoza wyniku meczu na bazie ratingu Elo i formy zespołów.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if team_a_name == team_b_name:
    st.warning("Wybierz dwie **różne** drużyny, aby zobaczyć prognozę meczu.")
    st.stop()

team_a = profiles[profiles["team"] == team_a_name].iloc[0]
team_b = profiles[profiles["team"] == team_b_name].iloc[0]

# ---- Matchup header -------------------------------------------------------

col_a, col_vs, col_b = st.columns([4, 1, 4])
with col_a:
    st.markdown(
        f"""
        <div class="matchup-card">
            <div class="subtitle">Drużyna A</div>
            <h2>🔵 {team_a_name}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_vs:
    st.markdown('<div class="vs-badge"><span>VS</span></div>', unsafe_allow_html=True)
with col_b:
    st.markdown(
        f"""
        <div class="matchup-card">
            <div class="subtitle">Drużyna B</div>
            <h2>🔴 {team_b_name}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Team form comparison --------------------------------------------------

st.markdown('<div class="section-title">📊 Aktualna Forma Drużyn</div>', unsafe_allow_html=True)

stat_col_a, stat_col_b, stat_col_c = st.columns(3)
elo_delta = team_a["current_elo"] - team_b["current_elo"]
poss_delta = team_a["Poss"] - team_b["Poss"]
gls_delta = team_a["Gls"] - team_b["Gls"]

stat_col_a.metric(
    label="Rating Elo (A vs B)",
    value=f"{team_a['current_elo']:.0f}",
    delta=f"{elo_delta:+.0f} vs {team_b_name}",
)
stat_col_b.metric(
    label="Posiadanie piłki % (A vs B)",
    value=f"{team_a['Poss']:.1f}%",
    delta=f"{poss_delta:+.1f} pp vs {team_b_name}",
)
stat_col_c.metric(
    label="Gole w turnieju (A vs B)",
    value=f"{int(team_a['Gls'])}",
    delta=f"{gls_delta:+.0f} vs {team_b_name}",
)

# ---- Prediction -------------------------------------------------------------

p_home, p_draw, p_away = predict_match(model, team_a, team_b)

favourite = team_a_name if p_home >= p_away and p_home >= p_draw else (
    team_b_name if p_away >= p_draw else "Remis"
)
favourite_prob = max(p_home, p_draw, p_away)

st.markdown(
    f"""
    <div class="verdict-banner">
        🔮 Model faworyzuje: <strong>{favourite}</strong>
        &nbsp;({favourite_prob * 100:.1f}% szans)
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">🎯 Prognozowane Prawdopodobieństwo Wyniku</div>', unsafe_allow_html=True)

metric_a, metric_draw, metric_b = st.columns(3)
metric_a.metric(f"🔵 Wygrana: {team_a_name}", f"{p_home * 100:.1f}%")
metric_draw.metric("⚪ Remis", f"{p_draw * 100:.1f}%")
metric_b.metric(f"🔴 Wygrana: {team_b_name}", f"{p_away * 100:.1f}%")

chart_df = pd.DataFrame(
    {
        "Wynik": [f"Wygrana {team_a_name}", "Remis", f"Wygrana {team_b_name}"],
        "Prawdopodobieństwo (%)": [
            round(p_home * 100, 1),
            round(p_draw * 100, 1),
            round(p_away * 100, 1),
        ],
    }
).set_index("Wynik")

st.bar_chart(chart_df, color="#1c7a4d")

st.caption(
    "⚠️ Prognozy generowane są przez model statystyczny (Random Forest) do celów "
    "demonstracyjnych/portfolio i nie stanowią porady bukmacherskiej."
)

with st.expander("🔧 Szczegóły techniczne modelu"):
    st.write(
        f"""
        - **Cechy modelu:** różnica ratingu Elo (`elo_diff`), skorygowana o różnicę
          posiadania piłki i liczby zdobytych goli w bieżącym turnieju
          (tzw. *Form Index*, patrz `notebooks/05_model_training.ipynb`).
        - **Algorytm:** `RandomForestClassifier(n_estimators=100, max_depth=5)`
          wytrenowany na {pd.read_csv(HISTORICAL_ELO_PATH).shape[0]:,} historycznych
          meczach międzynarodowych (`data/processed/historical_with_elo.csv`).
        - **Elo {team_a_name}:** {team_a['current_elo']:.1f} &nbsp;|&nbsp;
          **Elo {team_b_name}:** {team_b['current_elo']:.1f}
        - **Różnica Elo:** {elo_delta:+.1f}
        """
    )
