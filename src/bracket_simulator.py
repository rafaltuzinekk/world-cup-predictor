"""Symulator drabinki MŚ 2026 (1/16 finału -> Finał) z trybami "Time Machine" i Monte Carlo.

Punkt startowy wszystkich trybów to stan drużyn PO FAZIE GRUPOWEJ (przed 28.06.2026):
dopiero wtedy znane są pary 1/16 finału, więc to jest najwcześniejszy moment, w którym
da się prognozować drabinkę. Wyniki fazy pucharowej nigdy nie trafiają do Elo/formy
"od tyłu" - Elo i forma są aktualizowane dynamicznie w trakcie symulacji.

Tryby ``simulate_bracket``:
* ``"pre_tournament"`` - ignoruje ``KNOWN_RESULTS``; cała drabinka prognozowana modelem.
* ``"actual_plus_ai"`` - rozegrane mecze z ``KNOWN_RESULTS``, reszta prognozowana.
* ``"sandbox"``        - ``custom_overrides`` (wybory użytkownika), reszta liczona na żywo.

Rdzeń (``_simulate``) jest wektorowy po symulacjach, więc ta sama logika obsługuje
pojedynczą drabinkę deterministyczną (n=1) i 10 000 symulacji Monte Carlo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

from .database import BASE_DIR, get_connection
from .elo_engine import K_WORLD_CUP
from .ml_predictor import HOST_BONUS, STAGES, HybridMatchPredictor, host_flag

MONTE_CARLO_CSV = BASE_DIR / "data" / "processed" / "monte_carlo_results.csv"
MONTE_CARLO_TABLE = "monte_carlo_simulations"

# --------------------------------------------------------------------------- #
# Struktura drabinki (przeniesiona z app.py / notebooków 07-08)
# --------------------------------------------------------------------------- #
# Lewa połowa (8 meczów), potem prawa (8). Każda kolejna runda to "sparuj kolejnych zwycięzców".
ROUND_OF_32_FIXTURES: List[Tuple[str, str]] = [
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

# Mecze rozegrane w rzeczywistości; klucz = frozenset (kolejność drużyn nie ma znaczenia).
KNOWN_RESULTS: Dict[frozenset, str] = {
    frozenset({"Colombia", "Ghana"}): "Colombia",
    frozenset({"Argentina", "Cape Verde"}): "Argentina",
    frozenset({"Egypt", "Australia"}): "Egypt",
    frozenset({"Switzerland", "Algeria"}): "Switzerland",
    frozenset({"Portugal", "Croatia"}): "Portugal",
    frozenset({"Spain", "Austria"}): "Spain",
    frozenset({"United States", "Bosnia and Herzegovina"}): "United States",
    frozenset({"Belgium", "Senegal"}): "Belgium",
    frozenset({"England", "DR Congo"}): "England",
    frozenset({"Mexico", "Ecuador"}): "Mexico",
    frozenset({"France", "Sweden"}): "France",
    frozenset({"Norway", "Ivory Coast"}): "Norway",
    frozenset({"Morocco", "Netherlands"}): "Morocco",
    frozenset({"Paraguay", "Germany"}): "Paraguay",
    frozenset({"Brazil", "Japan"}): "Brazil",
    frozenset({"Canada", "South Africa"}): "Canada",
    frozenset({"Belgium", "United States"}): "Belgium",
    frozenset({"Spain", "Portugal"}): "Spain",
    frozenset({"England", "Mexico"}): "England",
    frozenset({"Norway", "Brazil"}): "Norway",
    frozenset({"France", "Paraguay"}): "France",
    frozenset({"Morocco", "Canada"}): "Morocco",
    frozenset({"Argentina", "Egypt"}): "Argentina",
    frozenset({"Switzerland", "Colombia"}): "Switzerland",
    frozenset({"France", "Morocco"}): "France",
    frozenset({"Spain", "Belgium"}): "Spain",
    frozenset({"England", "Norway"}): "England",
    frozenset({"Argentina", "Switzerland"}): "Argentina",
    frozenset({"Spain", "France"}): "Spain",
    frozenset({"Argentina", "England"}): "Argentina",
    frozenset({"Spain", "Argentina"}): "Spain",
}

ROUND_LABELS = ["1/16 Finału", "1/8 Finału", "Ćwierćfinał", "Półfinał", "Finał"]
MODES = ("pre_tournament", "actual_plus_ai", "sandbox")

TEAMS: List[str] = [team for fixture in ROUND_OF_32_FIXTURES for team in fixture]
TEAM_INDEX: Dict[str, int] = {team: i for i, team in enumerate(TEAMS)}
N_TEAMS = len(TEAMS)  # 32
N_ROUNDS = len(ROUND_LABELS)  # 5
assert len(STAGES) == N_ROUNDS

# Bonus gospodarza zależy od rundy (od ćwierćfinałów wszystkie mecze są w USA): (rundy, drużyny)
HOST_FLAGS_BY_ROUND = np.array([[host_flag(team, stage) for team in TEAMS] for stage in STAGES])

# Syntetyczny wynik meczu symulowanego/znanego bez bramek: zwycięzca 2:1 (G = 1.0 w Elo)
_SIM_GF_WIN, _SIM_GA_WIN = 2, 1

_DEFAULT_PREDICTOR: Optional[HybridMatchPredictor] = None


def get_default_predictor(con: Optional[duckdb.DuckDBPyConnection] = None) -> HybridMatchPredictor:
    """Leniwie tworzy i cache'uje domyślny predyktor (stan po fazie grupowej)."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        own = con is None
        con = con or get_connection(read_only=True)
        try:
            _DEFAULT_PREDICTOR = HybridMatchPredictor(con)
        finally:
            if own:
                con.close()
    return _DEFAULT_PREDICTOR


# --------------------------------------------------------------------------- #
# Wymuszone wyniki (KNOWN_RESULTS / custom_overrides)
# --------------------------------------------------------------------------- #
def _pair_key(key) -> frozenset:
    if isinstance(key, str):
        for sep in (" vs ", " | ", "|", " - "):
            if sep in key:
                return frozenset(p.strip() for p in key.split(sep, 1))
        raise ValueError(f"Nie rozpoznano pary drużyn w kluczu: {key!r}")
    pair = frozenset(key)
    if len(pair) != 2:
        raise ValueError(f"Klucz musi zawierać dokładnie 2 różne drużyny: {key!r}")
    return pair


def normalize_overrides(overrides: Optional[Mapping]) -> Dict[frozenset, str]:
    """Ujednolica ``custom_overrides``: klucze frozenset/tuple/list/'A vs B' -> frozenset."""
    result: Dict[frozenset, str] = {}
    for key, winner in (overrides or {}).items():
        pair = _pair_key(key)
        if winner not in pair:
            raise ValueError(f"Zwycięzca {winner!r} nie należy do pary {sorted(pair)}")
        result[pair] = winner
    return result


def _forced_matrix(forced: Mapping[frozenset, str]) -> np.ndarray:
    """Macierz T x T: [a, b] = indeks zwycięzcy meczu a-b albo -1."""
    matrix = np.full((N_TEAMS, N_TEAMS), -1, dtype=int)
    for pair, winner in forced.items():
        a, b = tuple(pair)
        if a in TEAM_INDEX and b in TEAM_INDEX:
            matrix[TEAM_INDEX[a], TEAM_INDEX[b]] = TEAM_INDEX[winner]
            matrix[TEAM_INDEX[b], TEAM_INDEX[a]] = TEAM_INDEX[winner]
    return matrix


def _resolve_forced(mode: str, custom_overrides: Optional[Mapping]) -> Tuple[Dict[frozenset, str], str]:
    if mode not in MODES:
        raise ValueError(f"Nieznany tryb {mode!r}; dostępne: {MODES}")
    if mode == "pre_tournament":
        return {}, "none"
    if mode == "actual_plus_ai":
        return dict(KNOWN_RESULTS), "real"
    return normalize_overrides(custom_overrides), "override"


# --------------------------------------------------------------------------- #
# Rdzeń symulacji (wektorowy po symulacjach)
# --------------------------------------------------------------------------- #
def _window_arrays(predictor: HybridMatchPredictor, n_sims: int, size: int = 8):
    """Okna ostatnich meczów (delta Elo, gf, ga) dla 32 drużyn: (S, T, size), NaN = brak meczu."""
    base = np.full((N_TEAMS, size, 3), np.nan)
    for team, idx in TEAM_INDEX.items():
        window = predictor.get_team_state(team)["window"][-size:]
        for j, entry in enumerate(reversed(window)):
            base[idx, size - 1 - j, :] = entry
    tiled = np.broadcast_to(base[None], (n_sims, N_TEAMS, size, 3)).copy()
    return tiled[..., 0], tiled[..., 1], tiled[..., 2]


def _window_mean(window: np.ndarray) -> np.ndarray:
    count = (~np.isnan(window)).sum(axis=-1)
    total = np.nansum(window, axis=-1)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def _simulate(
    predictor: HybridMatchPredictor,
    n_sims: int,
    forced: Mapping[frozenset, str],
    forced_kind: str,
    deterministic: bool,
    rng: Optional[np.random.Generator] = None,
    record: bool = False,
):
    """Symuluje ``n_sims`` drabinek. Zwraca (liczniki awansów (5, T), mecze lub None, finalne Elo)."""
    base_states = [predictor.get_team_state(t) for t in TEAMS]
    elo = np.tile(np.array([s["elo"] for s in base_states], dtype=float), (n_sims, 1))
    style = np.array([s["style_score"] for s in base_states], dtype=float)
    win_d, win_gf, win_ga = _window_arrays(predictor, n_sims)
    forced_mat = _forced_matrix(forced)
    has_forced = bool(forced)

    rows = np.arange(n_sims)[:, None]
    slots = np.tile(np.arange(N_TEAMS), (n_sims, 1))
    advance_counts = np.zeros((N_ROUNDS, N_TEAMS), dtype=np.int64)
    matches: List[dict] = []

    for rnd in range(N_ROUNDS):
        a, b = slots[:, 0::2], slots[:, 1::2]  # (S, m)
        elo_a, elo_b = elo[rows, a], elo[rows, b]
        form_a, form_b = _window_mean(win_d[rows, a]), _window_mean(win_d[rows, b])
        gf_a, ga_a_m = _window_mean(win_gf[rows, a]), _window_mean(win_ga[rows, a])
        gf_b, ga_b_m = _window_mean(win_gf[rows, b]), _window_mean(win_ga[rows, b])
        gd_a, gd_b = gf_a - ga_a_m, gf_b - ga_b_m

        host_r = HOST_FLAGS_BY_ROUND[rnd]
        host_a, host_b = host_r[a], host_r[b]
        pred = predictor.predict_arrays(
            elo_a, elo_b, host_a - host_b, form_a, form_b, gd_a, gd_b, style[a], style[b]
        )
        p_final = pred["p_final"]

        if deterministic:
            a_wins = p_final >= 0.5
        else:
            a_wins = rng.random(p_final.shape) < p_final

        is_forced = np.zeros(p_final.shape, dtype=bool)
        if has_forced:
            fw = forced_mat[a, b]
            is_forced = fw >= 0
            a_wins = np.where(is_forced, fw == a, a_wins)

        winners = np.where(a_wins, a, b)

        if record:
            for j in range(a.shape[1]):
                ia, ib, iw = int(a[0, j]), int(b[0, j]), int(winners[0, j])
                p_a = float(p_final[0, j])
                winner_prob = p_a if iw == ia else 1.0 - p_a
                forced_here = bool(is_forced[0, j])
                matches.append(
                    {
                        "round_index": rnd,
                        "round": ROUND_LABELS[rnd],
                        "stage": STAGES[rnd],
                        "match_index": j,
                        "team_a": TEAMS[ia],
                        "team_b": TEAMS[ib],
                        "winner": TEAMS[iw],
                        "prob_a": round(p_a * 100.0, 2),
                        "prob_b": round(100.0 - p_a * 100.0, 2),
                        "elo_prob_a": round(float(pred["p_elo"][0, j]) * 100.0, 2),
                        "elo_neutral_prob_a": round(float(pred["p_elo_neutral"][0, j]) * 100.0, 2),
                        "host_impact": round(float(pred["p_elo"][0, j] - pred["p_elo_neutral"][0, j]) * 100.0, 2),
                        "host_bonus_a": float(HOST_BONUS) if host_a[0, j] > host_b[0, j] else 0.0,
                        "host_bonus_b": float(HOST_BONUS) if host_b[0, j] > host_a[0, j] else 0.0,
                        "calibration_impact": round(float(pred["p_cal"][0, j] - pred["p_elo"][0, j]) * 100.0, 2),
                        "form_impact": round(float(pred["p_form"][0, j] - pred["p_cal"][0, j]) * 100.0, 2),
                        "style_impact": round(float(pred["p_final"][0, j] - pred["p_form"][0, j]) * 100.0, 2),
                        "elo_a": round(float(elo_a[0, j]), 1),
                        "elo_b": round(float(elo_b[0, j]), 1),
                        "form_a": round(float(form_a[0, j]), 2),
                        "form_b": round(float(form_b[0, j]), 2),
                        "gf_a": round(float(gf_a[0, j]), 2),
                        "ga_a": round(float(ga_a_m[0, j]), 2),
                        "gf_b": round(float(gf_b[0, j]), 2),
                        "ga_b": round(float(ga_b_m[0, j]), 2),
                        "is_real": forced_here and forced_kind == "real",
                        "is_override": forced_here and forced_kind == "override",
                        "is_upset": forced_here and winner_prob < 0.5,
                        "winner_prob": round(winner_prob * 100.0, 2),
                    }
                )

        # --- dynamiczna aktualizacja Elo i formy po rundzie (K = 60, G = 1, wynik 2:1) ---
        s_a = a_wins.astype(float)
        delta_a = K_WORLD_CUP * (s_a - pred["p_elo"])
        elo[rows, a] = elo_a + delta_a
        elo[rows, b] = elo_b - delta_a
        gf_a = np.where(a_wins, _SIM_GF_WIN, _SIM_GA_WIN).astype(float)
        ga_a = np.where(a_wins, _SIM_GA_WIN, _SIM_GF_WIN).astype(float)
        for team_idx, d, gf, ga in ((a, delta_a, gf_a, ga_a), (b, -delta_a, ga_a, gf_a)):
            for win, val in ((win_d, d), (win_gf, gf), (win_ga, ga)):
                current = win[rows, team_idx]  # (S, m, 8)
                win[rows, team_idx] = np.concatenate([current[..., 1:], val[..., None]], axis=-1)

        advance_counts[rnd] += np.bincount(winners.reshape(-1), minlength=N_TEAMS)
        slots = winners

    return advance_counts, (matches if record else None), elo


# --------------------------------------------------------------------------- #
# API publiczne
# --------------------------------------------------------------------------- #
def simulate_bracket(
    mode: str = "actual_plus_ai",
    custom_overrides: Optional[Mapping] = None,
    predictor: Optional[HybridMatchPredictor] = None,
    form_multiplier: Optional[float] = None,
    style_multiplier: Optional[float] = None,
) -> dict:
    """Deterministyczna drabinka (zawsze wygrywa faworyt modelu, o ile wynik nie jest wymuszony).

    ``form_multiplier`` / ``style_multiplier`` skalują wagi formy i stylu gry (1.0 = model
    bazowy). ``None`` = użyj wartości z predyktora (domyślnie 1.0).

    Zwraca słownik: ``mode``, ``rounds`` (lista rund, każda to lista meczów),
    ``champion``, ``finalists``, ``upset_count``, ``unmatched_known_results``.
    """
    forced, kind = _resolve_forced(mode, custom_overrides)
    predictor = (predictor or get_default_predictor()).with_multipliers(form_multiplier, style_multiplier)
    _, matches, _ = _simulate(predictor, 1, forced, kind, deterministic=True, record=True)

    rounds: List[List[dict]] = [[] for _ in range(N_ROUNDS)]
    for m in matches:
        rounds[m["round_index"]].append(m)

    played = {frozenset({m["team_a"], m["team_b"]}) for m in matches if m["is_real"] or m["is_override"]}
    final = rounds[-1][0]
    return {
        "mode": mode,
        "rounds": rounds,
        "champion": final["winner"],
        "finalists": (final["team_a"], final["team_b"]),
        "upset_count": sum(1 for m in matches if m["is_upset"]),
        "unmatched_known_results": sorted(
            tuple(sorted(p)) for p in set(forced) - played
        ),
    }


def prune_stale_overrides(
    custom_overrides: Optional[Mapping],
    predictor: Optional[HybridMatchPredictor] = None,
    form_multiplier: Optional[float] = None,
    style_multiplier: Optional[float] = None,
) -> Tuple[Dict[frozenset, str], dict]:
    """Usuwa nieaktualne nadpisania trybu Sandbox i zwraca ``(czyste_nadpisania, wynik_drabinki)``.

    Nadpisanie jest nieaktualne, gdy jego para drużyn nie występuje już w drabince - np. po
    zmianie zwycięzcy 1/16 finału mecz 1/8 finału ma innego rywala, więc ręczny wybór dla starej
    pary traci sens. Usunięcie nadpisania w jednej rundzie może zmienić pary w kolejnych,
    dlatego pętla powtarza symulację aż do ustabilizowania się drabinki.
    """
    overrides = normalize_overrides(custom_overrides)
    while True:
        result = simulate_bracket(
            "sandbox", overrides, predictor=predictor,
            form_multiplier=form_multiplier, style_multiplier=style_multiplier,
        )
        stale = {frozenset(pair) for pair in result["unmatched_known_results"]}
        if not stale:
            return overrides, result
        overrides = {pair: winner for pair, winner in overrides.items() if pair not in stale}


def run_monte_carlo(
    n_simulations: int = 10_000,
    mode: str = "pre_tournament",
    custom_overrides: Optional[Mapping] = None,
    seed: int = 42,
    predictor: Optional[HybridMatchPredictor] = None,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    save: bool = True,
    csv_path: Optional[Path] = MONTE_CARLO_CSV,
    form_multiplier: Optional[float] = None,
    style_multiplier: Optional[float] = None,
) -> pd.DataFrame:
    """Monte Carlo: % szans każdej z 32 drużyn na 1/8, 1/4, 1/2, Finał i tytuł.

    Prawdopodobieństwa pochodzą z ``HybridMatchPredictor`` i są liczone dynamicznie
    (Elo i forma aktualizowane po każdej rundzie w każdej symulacji).
    ``form_multiplier`` / ``style_multiplier``: jak w ``simulate_bracket``.
    """
    forced, kind = _resolve_forced(mode, custom_overrides)
    predictor = (predictor or get_default_predictor(con)).with_multipliers(form_multiplier, style_multiplier)
    rng = np.random.default_rng(seed)
    counts, _, _ = _simulate(predictor, n_simulations, forced, kind, deterministic=False, rng=rng)

    pct = counts / n_simulations * 100.0
    df = pd.DataFrame(
        {
            "team": TEAMS,
            "mode": mode,
            "n_simulations": n_simulations,
            "reach_round_of_16": pct[0],  # wygrana 1/16 = awans do 1/8
            "reach_quarter_final": pct[1],
            "reach_semi_final": pct[2],
            "reach_final": pct[3],
            "win_tournament": pct[4],
        }
    )
    df = df.sort_values("win_tournament", ascending=False).reset_index(drop=True)

    if save:
        own = con is None
        con = con or get_connection()
        try:
            con.register("mc_df", df)
            exists = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [MONTE_CARLO_TABLE]
            ).fetchone()[0]
            if exists:
                con.execute(f"DELETE FROM {MONTE_CARLO_TABLE} WHERE mode = ?", [mode])
                con.execute(f"INSERT INTO {MONTE_CARLO_TABLE} SELECT * FROM mc_df")
            else:
                con.execute(f"CREATE TABLE {MONTE_CARLO_TABLE} AS SELECT * FROM mc_df")
            con.unregister("mc_df")
        finally:
            if own:
                con.close()

        if csv_path is not None:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            out = pd.DataFrame(
                {  # pierwsze 4 kolumny = format z notebooka 07 (kompatybilność wsteczna)
                    "Team": df["team"],
                    "Win_Probability": df["win_tournament"].round(2),
                    "Reach_Final_Prob": df["reach_final"].round(2),
                    "Reach_Semi_Prob": df["reach_semi_final"].round(2),
                    "Reach_Quarter_Prob": df["reach_quarter_final"].round(2),
                    "Reach_R16_Prob": df["reach_round_of_16"].round(2),
                    "Mode": mode,
                    "N_Simulations": n_simulations,
                }
            )
            out.to_csv(csv_path, index=False)
    return df
