"""Pipeline (Kroki 1-2): dane -> Elo (od 1980) -> model hybrydowy -> drabinka -> Monte Carlo.

Uruchomienie:  python -m src.run_pipeline
"""
from __future__ import annotations

import math
import sys
import time

import pandas as pd

from .bracket_simulator import (
    KNOWN_RESULTS,
    TEAMS,
    run_monte_carlo,
    simulate_bracket,
)
from .database import DB_PATH, data_quality_report, init_database
from .elo_engine import CURRENT_ELO_CSV, DEFAULT_START_YEAR, EloEngine
from .ml_predictor import PRE_KNOCKOUT_CUTOFF, HybridMatchPredictor

FOCUS_TEAM = "Poland"
NEIGHBOURS = 3
N_SIMULATIONS = 10_000
SHOWCASE_PAIRS = [
    ("Germany", "Paraguay"),
    ("Netherlands", "Morocco"),
    ("Brazil", "Japan"),
]


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _fmt_ratings(df: pd.DataFrame) -> str:
    out = df[["rank", "team", "current_elo", "matches_played", "form_rating_change",
              "goals_scored_recent", "goals_conceded_recent"]].copy()
    out["current_elo"] = out["current_elo"].round(1)
    for col in ("form_rating_change", "goals_scored_recent", "goals_conceded_recent"):
        out[col] = out[col].round(2)
    return out.to_string(index=False)


def _rank_of(ratings: pd.DataFrame, team: str):
    row = ratings[ratings["team"] == team]
    return (int(row["rank"].iloc[0]), float(row["current_elo"].iloc[0])) if len(row) else (None, None)


def step_data():
    _section("1) BAZA DANYCH (DuckDB) + filtr FIFA")
    con = init_database()
    report = data_quality_report(con)
    print(f"Baza: {DB_PATH}")
    print(
        f"Mecze w CSV: {report['raw_rows_in_csv']} | zachowane (obie drużyny FIFA): "
        f"{report['matches_kept']} | odrzucone: {report['matches_dropped']}"
    )
    if report["fifa_members_without_matches"]:
        print(f"UWAGA - członkowie FIFA bez meczów: {report['fifa_members_without_matches']}")
    if report["fbref_teams_not_in_fifa"]:
        print(f"UWAGA - drużyny FBref nierozpoznane: {report['fbref_teams_not_in_fifa']}")
    return con


def step_elo(con) -> pd.DataFrame:
    _section(f"2) ELO z rozgrzewką od {DEFAULT_START_YEAR} r.")
    engine = EloEngine(start_year=DEFAULT_START_YEAR).fit_from_db(con)
    ratings = engine.save(con)
    print(f"Przetworzono {len(engine.history)} meczów | zapis: tabela elo_ratings + {CURRENT_ELO_CSV.name}")

    non_fifa = con.execute(
        "SELECT team FROM elo_ratings WHERE team NOT IN (SELECT team FROM fifa_members)"
    ).fetchall()
    assert not non_fifa, f"W rankingu są drużyny spoza FIFA: {non_fifa}"
    assert len(ratings) <= 211
    print(f"Drużyn w rankingu: {len(ratings)} (<= 211) | 'Ynys Môn': {bool((ratings['team'] == 'Ynys Môn').any())}")

    print("\nTOP 15 (Elo po rozgrzewce od 1980; forma = ostatnie 8 meczów)")
    print(_fmt_ratings(ratings.head(15)))

    old = EloEngine(start_year=2016).fit_from_db(con).ratings_table()
    new_rank, new_elo = _rank_of(ratings, FOCUS_TEAM)
    old_rank, old_elo = _rank_of(old, FOCUS_TEAM)
    print(f"\n{FOCUS_TEAM}: start 1980 -> miejsce {new_rank}/{len(ratings)}, Elo {new_elo:.1f}   "
          f"(start 2016 -> miejsce {old_rank}, Elo {old_elo:.1f})")
    i = int(ratings.index[ratings["team"] == FOCUS_TEAM][0])
    print(_fmt_ratings(ratings.iloc[max(0, i - NEIGHBOURS): i + NEIGHBOURS + 1]))
    return ratings


def step_model(con) -> HybridMatchPredictor:
    _section(f"3) HYBRYDOWY MODEL ML (stan po fazie grupowej, mecze < {PRE_KNOCKOUT_CUTOFF})")
    t0 = time.time()
    predictor = HybridMatchPredictor(con)
    c, d = predictor.coefficients, predictor.diagnostics
    print(f"Trening: {predictor.n_training_matches} meczów (od 1990, Elo/forma liczone PRZED meczem) | {time.time() - t0:.1f}s")
    print(
        "Współczynniki LogisticRegression: "
        f"elo_logit={c['elo_logit']:.3f}  form_diff={c['form_diff']:+.3f}  goal_balance_diff={c['goal_balance_diff']:+.3f}"
    )
    if d:
        print(
            f"Walidacja czasowa (trening <2022, test {d['holdout_matches']} meczów >=2022), log-loss (niżej = lepiej): "
            f"czyste Elo {d['logloss_elo']:.4f} | ensemble bez formy {d['logloss_ensemble_no_form']:.4f} | "
            f"ensemble z formą {d['logloss_ensemble_with_form']:.4f}"
        )
    return predictor


def step_pairs(predictor: HybridMatchPredictor) -> None:
    _section("4) Czyste Elo vs model hybrydowy - wybrane pary 1/16 finału")
    for a, b in SHOWCASE_PAIRS:
        r = predictor.predict_match(a, b)
        real = KNOWN_RESULTS.get(frozenset({a, b}), "-")
        print(f"\n{a} vs {b}   (rzeczywisty zwycięzca: {real})")
        print(f"  Czyste Elo : {a} {r['elo_prob_a']:5.1f}% | {b} {r['elo_prob_b']:5.1f}%   (Elo {r['elo_a']:.0f} vs {r['elo_b']:.0f})")
        print(f"  Hybryda    : {a} {r['prob_a']:5.1f}% | {b} {r['prob_b']:5.1f}%")
        print(
            f"  Wpływ (pp dla {a}): gospodarz {r['host_impact']:+.2f} | kalibracja {r['calibration_impact']:+.2f} | "
            f"forma {r['form_impact']:+.2f} | styl {r['style_impact']:+.2f}"
        )
        print(f"  {r['summary']}")


def _backtest(result: dict) -> str:
    real = [m for rnd in result["rounds"] for m in rnd if m["is_real"]]
    if not real:
        return "brak rozegranych meczów"

    def nll(prob_pct: float) -> float:
        return -math.log(max(prob_pct, 0.1) / 100.0)

    elo_ll, hyb_ll = [], []
    for m in real:
        elo_w = m["elo_prob_a"] if m["winner"] == m["team_a"] else 100.0 - m["elo_prob_a"]
        elo_ll.append(nll(elo_w))
        hyb_ll.append(nll(m["winner_prob"]))
    n = len(real)
    elo_acc = sum(1 for m in real if (m["elo_prob_a"] >= 50) == (m["winner"] == m["team_a"])) / n
    hyb_acc = sum(1 for m in real if m["winner_prob"] >= 50) / n
    return (
        f"n={n} rozegranych meczów (dynamiczne Elo/forma, bez dostrajania do tych wyników) | "
        f"trafność: Elo {elo_acc:.0%} vs hybryda {hyb_acc:.0%} | "
        f"log-loss: Elo {sum(elo_ll) / n:.4f} vs hybryda {sum(hyb_ll) / n:.4f}"
    )


def step_brackets(predictor: HybridMatchPredictor) -> None:
    _section("5) SYMULATOR DRABINKI (deterministyczny)")
    pre = simulate_bracket("pre_tournament", predictor=predictor)
    act = simulate_bracket("actual_plus_ai", predictor=predictor)

    for res, label in ((pre, "pre_tournament  (czysta prognoza od 1/16)"),
                       (act, "actual_plus_ai  (wyniki rzeczywiste + AI)")):
        fa, fb = res["finalists"]
        print(f"\n[{label}]")
        print(f"  Finał: {fa} vs {fb}  ->  MISTRZ ŚWIATA: {res['champion']}   | niespodzianki: {res['upset_count']}")
        if res["unmatched_known_results"]:
            print(f"  Wpisy KNOWN_RESULTS bez pary w drabince: {res['unmatched_known_results']}")

    print("\nPrzebieg pre_tournament (zwycięzcy kolejnych rund):")
    for rnd in pre["rounds"]:
        print(f"  {rnd[0]['round']:<12}: " + ", ".join(m["winner"] for m in rnd))

    upsets = [m for rnd in act["rounds"] for m in rnd if m["is_upset"]]
    print(f"\nNiespodzianki w actual_plus_ai (zwycięzca miał <50% wg modelu hybrydowego): {len(upsets)}")
    for m in upsets:
        loser = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
        print(f"  {m['round']:<12} {m['winner']} pokonał {loser} (miał {m['winner_prob']:.1f}%)")

    print(f"\nBacktest modelu na meczach rzeczywistych: {_backtest(act)}")


def step_monte_carlo(con, predictor: HybridMatchPredictor) -> None:
    _section(f"6) MONTE CARLO - {N_SIMULATIONS} symulacji (pre_tournament)")
    t0 = time.time()
    mc = run_monte_carlo(N_SIMULATIONS, mode="pre_tournament", predictor=predictor, con=con)
    print(f"Czas: {time.time() - t0:.1f}s | zapis: tabela monte_carlo_simulations + monte_carlo_results.csv")
    assert len(mc) == len(TEAMS) == 32 and abs(mc["win_tournament"].sum() - 100.0) < 1e-6

    top = mc.head(5).copy()
    top = top.rename(
        columns={
            "reach_round_of_16": "1/8 %",
            "reach_quarter_final": "1/4 %",
            "reach_semi_final": "1/2 %",
            "reach_final": "Finał %",
            "win_tournament": "Mistrz %",
        }
    )
    cols = ["team", "1/8 %", "1/4 %", "1/2 %", "Finał %", "Mistrz %"]
    print("\nTOP 5 faworytów:")
    print(top[cols].round(1).to_string(index=False))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== KROK 2: Elo (1980) + hybrydowy model + drabinka + Monte Carlo ===")
    con = step_data()
    try:
        step_elo(con)
        predictor = step_model(con)
        step_pairs(predictor)
        step_brackets(predictor)
        step_monte_carlo(con, predictor)
    finally:
        con.close()


if __name__ == "__main__":
    main()
