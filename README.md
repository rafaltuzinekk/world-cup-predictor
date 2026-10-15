# FIFA World Cup 2026: Knockout-Stage Prediction Engine

An end-to-end analytics project: a local **DuckDB** warehouse, a dynamic **Elo** rating engine, a **hybrid ML match model**
(Elo + recent form + playing style) and an interactive **Streamlit** dashboard with a clickable what-if bracket and a
vectorised **Monte Carlo** simulator, packaged with **Docker** and **GitHub Actions**.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-SQL%20%2F%20OLAP-FFF000?logo=duckdb&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-LogReg%20%2B%20RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20charts-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Custom%20Components-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%20%2B%20Keep--Alive-2088FF?logo=githubactions&logoColor=white)

**Live demo:** <https://world-cup-predictor-rt.streamlit.app/>

## What it does

| Dashboard tab | What you get |
|---|---|
| **Knockout bracket** | Full Round of 32 → Final bracket with KPIs. Three modes: *Actual + AI* (played matches use real results), *Pre-tournament* (100% model forecast) and *Sandbox*, where you **click a team inside any match card** to advance it. The rest of the bracket is recomputed instantly. Clicking the selected (purple) team again reverts to the AI pick. |
| **Head-to-Head** | Any two nations, with a waterfall chart that decomposes the win probability (Elo → host bonus → calibration → form → style), a percentile radar and a stats table. |
| **Monte Carlo** | 10,000 simulated tournaments (live recomputation with your current mode and weights) and title odds for all 32 teams. |
| **World Elo ranking** | 211 FIFA members with form, goal balance, search and filters. |

## System architecture (`src/`)

```
data/raw/results.csv ─┐
data/processed/       │   1. database.py ──► 2. elo_engine.py ──► 3. ml_predictor.py ──► 4. bracket_simulator.py ──► app.py
 fbref_group_stats.csv┘   DuckDB + FIFA      dynamic Elo           HybridMatchPredictor    Time Machine / Sandbox / MC    Streamlit UI
                          filter (211)       (1980-2026)           Elo + form + style      + bracket_component/ (JS)
```

1. **`src/database.py` - data layer.** A local OLAP database in DuckDB. Raw results (49,494 matches) are loaded with SQL and
   filtered to matches where **both teams are among the 211 official FIFA members**. This removes 3,999 anomalous rows
   (non-FIFA sides such as Ynys Môn, Jersey, Padania or Abkhazia) and leaves 45,495 matches. The module also normalises
   team-name aliases (e.g. FBref → `results.csv` naming) and produces a data-quality report.
2. **`src/elo_engine.py` - rating engine.** A dynamic Elo model with a **warm-up from 1980** that fixes the cold-start
   problem of earlier versions (Poland: rank 57 → 44). Tournament-dependent `K` (10 friendlies … 60 World Cup, 40 continental
   finals, 25 qualifiers / Nations League), a goal-difference multiplier `G` (1.0 for 0-1 goals, 1.5 for 2, `(11+N)/8` for 3+)
   and a home advantage of +100 for non-neutral matches. For the **2026 World Cup, the host bonus (+75 Elo) depends on the
   round**: the USA get it in every round, Mexico and Canada only in the Round of 32 and Round of 16, because all later
   matches are played in the USA. A `cutoff_date` option reproduces the state of any team "as of day X" without leakage.
3. **`src/ml_predictor.py` - `HybridMatchPredictor`.** Three components combined in logit space:
   - **Base Elo** (with round-aware host bonus),
   - **Form** from each team's last 8 matches (Elo change per match and goal balance): a `LogisticRegression` +
     `RandomForest` ensemble blended with pure Elo. Features are computed strictly *before* each historical match and the
     model is validated on a temporal hold-out (train < 2022, test ≥ 2022), so there is no data leakage,
   - **Playing style** from FBref group-stage data (possession, non-penalty goals per 90, finishing above what possession
     implies), applied as a **capped** logit correction (`STYLE_WEIGHT`, `STYLE_CAP`) because there is no historical sample
     to fit it on.

   `form_multiplier` / `style_multiplier` (exposed as sliders in the UI) rescale each component without retraining.
4. **`src/bracket_simulator.py` & `app.py` - simulation and UI.**
   - A deterministic **"Time Machine"** simulator (`pre_tournament`, `actual_plus_ai`) where the favourite always advances
     unless the result is forced. Elo and form are updated dynamically after every round.
   - A **`sandbox`** mode driven by a **bidirectional Streamlit custom component** (`src/bracket_component/index.html`,
     plain JS/HTML, no build step): the bracket is drawn as HTML/CSS inside an iframe and every click on a team row is sent
     back to Python. Stale manual picks on later rounds are pruned automatically when an earlier winner changes
     (`prune_stale_overrides`).
   - A **vectorised Monte Carlo engine**: the same core runs 10,000 tournaments with NumPy in about 4-7 seconds.

## Key analytical insights

**Why the deterministic bracket and Monte Carlo name different favourites.**
In the pre-tournament forecast the deterministic bracket crowns **France** (Final: France vs Argentina), but Monte Carlo
gives **Argentina the highest title odds (24.5% vs 20.4% for France)**. Both are correct; they answer different questions:

- The deterministic bracket takes the *most likely outcome of every single match*. France beats Argentina in the final with
  just **51.3%**, and the chain of "favourite always wins" gives France the trophy.
- Monte Carlo integrates over **path risk**. Argentina has the best rating (2150 Elo) and sits in the half of the draw with
  Brazil (6.8% title odds), while France's half contains Spain (11.9%). Argentina therefore reaches the final in 41.3% of
  simulations versus 33.6% for France. Title odds are *P(reach the final) × P(win it)*, and a near coin-flip in the final
  does not offset a gap of almost 8 points in reaching it.
- The deterministic bracket contains no upsets by construction (unless a result is forced), whereas Monte Carlo shows how
  fragile favourites really are: even Argentina wins the tournament in only 24.5% of simulations.

**How the Elo + Form + Style decomposition works.** Every probability is shown as a chain of corrections to pure Elo
(percentage points for the first team). Example from the Round of 32, **Germany vs Paraguay**:

| Step | Germany win probability |
|---|---|
| Pure Elo (1921 vs 1807) | 65.9% |
| Host bonus | +0.0 pp |
| Re-calibration | −1.4 pp |
| Recent form (last 8 matches) | +3.2 pp |
| Playing style (possession, goals/90, finishing) | +6.2 pp |
| **Hybrid model** | **73.8%** |

For **Netherlands vs Morocco** the same chain barely moves the needle (55.3% → 57.8%). In the dashboard each match tooltip
and the Head-to-Head waterfall show which factor drives the difference. Paraguay actually won that match as a 26.2% underdog;
this is exactly what the "upset" marker highlights.

**Honest model assessment.**
- Temporal validation (4,412 matches since 2022), log-loss: pure Elo **0.5506**, ensemble with form **0.5500**. Form
  adds only a marginal improvement, which is why its influence is learned from data and not hand-tuned.
- Playing style cannot be validated historically, so it is deliberately capped.
- On the 31 knockout matches already played, accuracy is 77% for both Elo and the hybrid model (log-loss 0.476 vs 0.482),
  too small a sample to claim an edge. The model is a decision-support tool, not betting advice.
- Elo with a 1980 warm-up favours teams that play many games in weaker confederations (e.g. Mexico).

## How to run

### Option A - Docker

```bash
docker compose up --build
# open http://localhost:8501
```

The `world-cup-analytics` service maps port `8501` and mounts `./data` into the container, so the DuckDB database persists
between runs. On start, the container checks for `data/world_cup.duckdb` and, if it is missing, runs
`python -m src.run_pipeline` before launching Streamlit.

### Option B - local Python

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_pipeline      # builds data/world_cup.duckdb, Elo ratings, Monte Carlo CSV (~20 s)
streamlit run app.py
```

The dashboard also rebuilds the database automatically on first launch if it is missing.

## Automation (GitHub Actions)

`.github/workflows/ci_and_keepalive.yml`:

- **CI Pipeline Check** - on every push to `main`: installs dependencies, runs `python -m src.run_pipeline`, verifies the
  generated artifacts and renders `app.py` headlessly with Streamlit's `AppTest`.
- **Keep-Alive** - every 12 hours (and on manual `workflow_dispatch`): an HTTP request with browser headers to the Streamlit
  Community Cloud app, to reduce hibernation of the free-tier deployment.

## Repository layout

```
app.py                      Streamlit dashboard
src/
  database.py               DuckDB layer, FIFA filter, data-quality report
  elo_engine.py             Dynamic Elo engine
  ml_predictor.py           HybridMatchPredictor
  bracket_simulator.py      Bracket simulator + Monte Carlo
  bracket_component/        Bidirectional JS/HTML Streamlit component
  flags.py                  Team → flag helpers
  run_pipeline.py           End-to-end pipeline with console report
data/
  raw/results.csv           Historical international results
  processed/                Elo ratings, FBref stats, Monte Carlo output
  world_cup.duckdb          Pre-built database (rebuildable)
notebooks/                  Original research notebooks (archive)
Dockerfile, docker-compose.yml, .github/workflows/
```

## Data sources

International results (Kaggle: *International football results from 1872*) and group-stage team statistics from FBref.
