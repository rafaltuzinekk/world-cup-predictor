"""Hybrydowy model predykcji meczu pucharowego: Elo + Forma + Styl gry (FBref).

Trzy filary:

a) **Siła bazowa (Elo)** - probabilistyczny model Elo z bonusem +75 pkt dla gospodarzy
   MŚ 2026 w fazie pucharowej. Od ćwierćfinałów wszystkie mecze rozgrywane są w USA, więc
   bonus przysługuje: United States - we wszystkich rundach, Mexico i Canada - tylko
   w 1/16 (R32) i 1/8 (R16) finału.
b) **Forma** - różnica ``form_rating_change`` (średnia zmiana Elo z 8 ostatnich meczów)
   oraz bilansu bramkowego z 8 ostatnich meczów. Wpływ formy jest WYUCZONY:
   ``LogisticRegression`` + ``RandomForestClassifier`` trenowane na historii meczów
   (cechy liczone PRZED meczem, brak wycieku danych), a następnie łączone w ensemble
   z czystym Elo.
c) **Styl gry (FBref)** - złożony wskaźnik z posiadania piłki, goli (bez karnych) na 90 min
   oraz skuteczności ponad poziom wynikający z posiadania. Dla stylu nie istnieje
   historyczna próbka (statystyki FBref mamy tylko dla fazy grupowej MŚ 2026), więc
   działa on jako *skalibrowana, ograniczona korekta logitu* (``STYLE_WEIGHT``,
   ``STYLE_CAP``), a nie jako wyuczony parametr.

Rozkład różnicy względem Elo bez bonusu gospodarza (w punktach procentowych, z perspektywy
drużyny A)::

    prob_a - elo_neutral_prob_a = host_impact + calibration_impact + form_impact + style_impact

``calibration_impact`` to efekt samej rekalibracji Elo przez ensemble (bez formy i stylu).
Wagi formy i stylu można skalować mnożnikami ``form_multiplier`` / ``style_multiplier``
(1.0 = model bazowy, 0.0 = wyłączenie danego czynnika).
"""
from __future__ import annotations

import copy
import math
from typing import Dict, Mapping, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .elo_engine import DEFAULT_START_YEAR, HOME_ADVANTAGE, EloEngine

# --------------------------------------------------------------------------- #
# Parametry
# --------------------------------------------------------------------------- #
HOSTS_2026 = frozenset({"United States", "Mexico", "Canada"})
HOST_BONUS = 75.0  # pkt Elo dla gospodarza MŚ 2026 w meczu pucharowym rozgrywanym u siebie

# Rundy fazy pucharowej (kolejność = kolejność rund w drabince).
STAGES = ("R32", "R16", "QF", "SF", "F")
_ALL_STAGES = frozenset(STAGES)
# Od ćwierćfinałów wszystkie mecze odbywają się w USA - Mexico i Canada grają "u siebie"
# wyłącznie w 1/16 (R32) i 1/8 (R16) finału.
HOST_BONUS_STAGES = {
    "United States": _ALL_STAGES,
    "Mexico": frozenset({"R32", "R16"}),
    "Canada": frozenset({"R32", "R16"}),
}


def host_flag(team: str, stage: str) -> float:
    """1.0, jeśli ``team`` jest gospodarzem MŚ 2026 z bonusem w rundzie ``stage``, inaczej 0.0."""
    return 1.0 if stage in HOST_BONUS_STAGES.get(team, ()) else 0.0



# Pierwszy dzień fazy pucharowej MŚ 2026: stan drużyn "przed drabinką" = po fazie grupowej.
PRE_KNOCKOUT_CUTOFF = "2026-06-28"

TRAIN_FROM_YEAR = 1990  # 1980-1989 to burn-in Elo, nie trenujemy na nim modeli
HOLDOUT_YEAR = 2022  # walidacja czasowa: trening < 2022, test >= 2022
FORM_SCALE = 10.0  # różnica formy (pkt Elo/mecz) skalowana do rzędu jedności
ENSEMBLE_ELO_WEIGHT = 0.40  # udział czystego Elo w ensemble (reszta: LR + RF)
STYLE_WEIGHT = 0.15  # zmiana logitu na 1 odchylenie standardowe różnicy stylu
STYLE_CAP = 0.30  # maks. |zmiana logitu| od stylu (~ +-7 pp przy 50%)
STYLE_COMPONENT_WEIGHTS = {"attack": 0.40, "possession": 0.35, "clinical": 0.25}

LN10_OVER_400 = math.log(10.0) / 400.0
_EPS = 1e-6

TeamState = Dict[str, object]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    return (values - values.mean()) / std if std > 0 else np.zeros_like(values)


class HybridMatchPredictor:
    """Ensemble: probabilistyczne Elo + (LogisticRegression, RandomForest) na formie + styl FBref."""

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        cutoff_date: str = PRE_KNOCKOUT_CUTOFF,
        start_year: int = DEFAULT_START_YEAR,
        random_state: int = 42,
        evaluate: bool = True,
        form_multiplier: float = 1.0,
        style_multiplier: float = 1.0,
    ) -> None:
        self.con = con
        self.cutoff_date = cutoff_date
        self.start_year = start_year
        self.random_state = random_state
        self.form_multiplier = float(form_multiplier)
        self.style_multiplier = float(style_multiplier)

        self.engine: Optional[EloEngine] = None
        self.team_state: Dict[str, TeamState] = {}
        self.style_table: pd.DataFrame = pd.DataFrame()
        self.diagnostics: Dict[str, float] = {}
        self.coefficients: Dict[str, float] = {}

        self._fit(evaluate=evaluate)

    # ------------------------------------------------------------------ fit
    def _fit(self, evaluate: bool) -> None:
        self.engine = EloEngine(start_year=self.start_year).fit_from_db(
            self.con, cutoff_date=self.cutoff_date
        )
        self._build_style_table()
        self._build_team_state()

        history = self.engine.history_table()
        history = history[history["date"].dt.year >= TRAIN_FROM_YEAR]
        X, y, w = self._training_set(history)
        self.models = self._fit_models(X, y, w)
        self.coefficients = {
            "elo_logit": float(self.models[0].coef_[0][0]),
            "form_diff": float(self.models[0].coef_[0][1]),
            "goal_balance_diff": float(self.models[0].coef_[0][2]),
        }
        self.n_training_matches = int(len(history))

        if evaluate:
            self._evaluate_holdout(history)

    def with_multipliers(
        self,
        form_multiplier: Optional[float] = None,
        style_multiplier: Optional[float] = None,
    ) -> "HybridMatchPredictor":
        """Płytka kopia z innymi mnożnikami wag (bez ponownego treningu modeli).

        ``None`` oznacza "zostaw obecną wartość". Jeśli nic się nie zmienia, zwraca ``self``.
        """
        fm = self.form_multiplier if form_multiplier is None else float(form_multiplier)
        sm = self.style_multiplier if style_multiplier is None else float(style_multiplier)
        if fm == self.form_multiplier and sm == self.style_multiplier:
            return self
        clone = copy.copy(self)
        clone.form_multiplier = fm
        clone.style_multiplier = sm
        return clone

    def _build_style_table(self) -> None:
        stats = self.con.execute("SELECT * FROM team_stats_fbref").df()
        poss = stats["Poss"].astype(float).to_numpy()
        gls90 = stats["Gls.1"].astype(float).to_numpy()
        npg90 = stats["G-PK.1"].astype(float).to_numpy()

        # skuteczność ponad posiadanie: reszta z regresji (gole/90 ~ posiadanie)
        slope, intercept = np.polyfit(poss, npg90, 1)
        clinical = npg90 - (slope * poss + intercept)

        cw = STYLE_COMPONENT_WEIGHTS
        raw = (
            cw["attack"] * _zscore(npg90)
            + cw["possession"] * _zscore(poss)
            + cw["clinical"] * _zscore(clinical)
        )
        style = _zscore(raw)  # ponowna standaryzacja: 1.0 = 1 odchylenie std. wśród 48 drużyn

        self.style_table = pd.DataFrame(
            {
                "team": stats["team"],
                "possession": poss,
                "goals_per90": gls90,
                "npg_per90": npg90,
                "clinical_vs_possession": clinical,
                "style_score": style,
            }
        ).set_index("team")

    def _build_team_state(self) -> None:
        assert self.engine is not None
        for team, rating in self.engine.ratings.items():
            if self.engine.matches_played[team] <= 0:
                continue
            self.team_state[team] = self._state_from_engine(team, rating)

    def _state_from_engine(self, team: str, rating: float) -> TeamState:
        assert self.engine is not None
        form = self.engine.team_form(team)
        style = self.style_table.loc[team] if team in self.style_table.index else None
        return {
            "elo": float(rating),
            "form": form["form_rating_change"],
            "goals_scored": form["goals_scored_recent"],
            "goals_conceded": form["goals_conceded_recent"],
            "recent_form": self.engine.recent_form(team),
            "window": self.engine.get_form_window(team),
            "possession": float(style["possession"]) if style is not None else float("nan"),
            "goals_per90": float(style["goals_per90"]) if style is not None else float("nan"),
            "style_score": float(style["style_score"]) if style is not None else 0.0,
            "is_host": team in HOSTS_2026,
        }

    def get_team_state(self, team: str) -> TeamState:
        """Stan drużyny po fazie grupowej (Elo, forma, styl). Nieznana drużyna -> Elo 1500."""
        if team in self.team_state:
            return self.team_state[team]
        return self._state_from_engine(team, self.engine.base_rating)  # type: ignore[union-attr]

    # --------------------------------------------------------------- trening
    @staticmethod
    def _features(elo_diff, form_diff, gd_diff) -> np.ndarray:
        return np.column_stack(
            [
                np.asarray(elo_diff, dtype=float) * LN10_OVER_400,
                np.asarray(form_diff, dtype=float) / FORM_SCALE,
                np.asarray(gd_diff, dtype=float),
            ]
        )

    def _training_set(self, history: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zbiór treningowy z antysymetryczną augmentacją i miękkim celem (remis = 0.5).

        Dla meczu z wynikiem s (1 / 0.5 / 0 z perspektywy gospodarza) tworzymy wiersze
        (X, y=1, w=s), (X, y=0, w=1-s) oraz lustrzane (-X, y=1, w=1-s), (-X, y=0, w=s).
        Dzięki temu model szacuje oczekiwany wynik (jak Elo), a nie samo P(zwycięstwo
        w 90 min) - co jest właściwe dla fazy pucharowej, gdzie remis rozstrzygają
        dogrywka i karne.
        """
        elo_diff = (
            history["home_elo_pre"].to_numpy()
            - history["away_elo_pre"].to_numpy()
            + np.where(history["neutral"].to_numpy(), 0.0, HOME_ADVANTAGE)
        )
        X = self._features(
            elo_diff,
            history["home_form_pre"].to_numpy() - history["away_form_pre"].to_numpy(),
            history["home_gd_pre"].to_numpy() - history["away_gd_pre"].to_numpy(),
        )
        hs, as_ = history["home_score"].to_numpy(), history["away_score"].to_numpy()
        s = np.where(hs > as_, 1.0, np.where(hs == as_, 0.5, 0.0))

        Xs = np.vstack([X, X, -X, -X])
        ys = np.concatenate([np.ones_like(s), np.zeros_like(s), np.ones_like(s), np.zeros_like(s)])
        ws = np.concatenate([s, 1.0 - s, 1.0 - s, s])
        keep = ws > 0
        return Xs[keep], ys[keep], ws[keep]

    def _fit_models(self, X, y, w) -> Tuple[LogisticRegression, RandomForestClassifier]:
        # fit_intercept=False => P(A vs B) + P(B vs A) = 1 (brak wbudowanej przewagi "gospodarza")
        lr = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000)
        lr.fit(X, y, sample_weight=w)
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_leaf=100,
            random_state=self.random_state,
            n_jobs=1,
        )
        rf.fit(X, y, sample_weight=w)
        return lr, rf

    # -------------------------------------------------------------- predykcja
    @staticmethod
    def _ml_prob(models, X: np.ndarray) -> np.ndarray:
        """Średnia LR i RF, symetryzowana: p(X) = (f(X) + 1 - f(-X)) / 2."""
        lr, rf = models

        def f(Z):
            return 0.5 * lr.predict_proba(Z)[:, 1] + 0.5 * rf.predict_proba(Z)[:, 1]

        return 0.5 * (f(X) + 1.0 - f(-X))

    def _blend(self, p_elo: np.ndarray, X: np.ndarray, models=None) -> np.ndarray:
        p_ml = self._ml_prob(models or self.models, X)
        return ENSEMBLE_ELO_WEIGHT * p_elo + (1.0 - ENSEMBLE_ELO_WEIGHT) * p_ml

    def predict_arrays(
        self,
        elo_a, elo_b, host_diff, form_a, form_b, gd_a, gd_b, style_a, style_b,
        form_multiplier: Optional[float] = None,
        style_multiplier: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Wektorowa predykcja (dowolny kształt wejścia; zwracane tablice mają kształt wejścia).

        ``host_diff`` = (flaga bonusu gospodarza A) - (flaga B), patrz ``host_flag``.
        ``form_multiplier`` / ``style_multiplier`` skalują wpływ formy i stylu (domyślnie
        wartości zapisane w predyktorze, czyli 1.0).

        Zwraca prawdopodobieństwa (0-1) zwycięstwa/awansu drużyny A:
        ``p_elo_neutral`` (Elo bez bonusu gospodarza), ``p_elo`` (Elo z bonusem),
        ``p_cal`` (ensemble bez formy), ``p_form`` (ensemble z formą), ``p_final`` (po korekcie stylem).
        """
        fm = self.form_multiplier if form_multiplier is None else float(form_multiplier)
        sm = self.style_multiplier if style_multiplier is None else float(style_multiplier)
        shape = np.shape(elo_a)
        flat = lambda v: np.asarray(v, dtype=float).reshape(-1)  # noqa: E731
        elo_a, elo_b, host_diff = flat(elo_a), flat(elo_b), flat(host_diff)
        form_a, form_b, gd_a, gd_b = flat(form_a), flat(form_b), flat(gd_a), flat(gd_b)
        style_a, style_b = flat(style_a), flat(style_b)

        elo_diff_neutral = elo_a - elo_b
        elo_diff = elo_diff_neutral + HOST_BONUS * host_diff
        X_full = self._features(elo_diff, form_a - form_b, gd_a - gd_b)
        X_zero = X_full.copy()
        X_zero[:, 1:] = 0.0

        p_elo_neutral = _sigmoid(elo_diff_neutral * LN10_OVER_400)
        p_elo = _sigmoid(X_full[:, 0])
        p_cal = self._blend(p_elo, X_zero)
        p_form_base = self._blend(p_elo, X_full)
        if fm == 1.0:
            p_form = p_form_base
        else:
            # skalowanie korekty formy w przestrzeni logitu (0.0 = brak formy, 1.0 = model bazowy)
            p_form = _sigmoid(_logit(p_cal) + fm * (_logit(p_form_base) - _logit(p_cal)))
        shift = np.clip(STYLE_WEIGHT * (style_a - style_b), -STYLE_CAP, STYLE_CAP) * sm
        p_final = np.clip(_sigmoid(_logit(p_form) + shift), 0.001, 0.999)

        return {
            "p_elo_neutral": p_elo_neutral.reshape(shape),
            "p_elo": p_elo.reshape(shape),
            "p_cal": p_cal.reshape(shape),
            "p_form": p_form.reshape(shape),
            "p_final": p_final.reshape(shape),
        }

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        state: Optional[Mapping[str, TeamState]] = None,
        stage: str = "R32",
    ) -> dict:
        """Szanse awansu w meczu fazy pucharowej.

        ``state`` (opcjonalnie): słownik ``{drużyna: stan}`` nadpisujący bazowy stan
        (np. Elo/formę po wcześniejszych rundach). Domyślnie stan po fazie grupowej.
        ``stage``: runda (``STAGES``) - decyduje o tym, czy gospodarzom MŚ 2026 przysługuje
        bonus (Mexico i Canada tylko w R32/R16, United States zawsze).
        """
        if stage not in STAGES:
            raise ValueError(f"Nieznana runda {stage!r}; dostępne: {STAGES}")
        sa = (state or {}).get(team_a) or self.get_team_state(team_a)
        sb = (state or {}).get(team_b) or self.get_team_state(team_b)
        host_diff = host_flag(team_a, stage) - host_flag(team_b, stage)

        out = self.predict_arrays(
            [sa["elo"]], [sb["elo"]], [host_diff],
            [sa["form"]], [sb["form"]],
            [float(sa["goals_scored"]) - float(sa["goals_conceded"])],
            [float(sb["goals_scored"]) - float(sb["goals_conceded"])],
            [sa["style_score"]], [sb["style_score"]],
        )
        p_neutral, p_elo, p_cal, p_form, p_final = (
            float(out[k][0]) for k in ("p_elo_neutral", "p_elo", "p_cal", "p_form", "p_final")
        )

        prob_a = round(p_final * 100.0, 2)
        result = {
            "team_a": team_a,
            "team_b": team_b,
            "stage": stage,
            "prob_a": prob_a,
            "prob_b": round(100.0 - prob_a, 2),
            "elo_prob_a": round(p_elo * 100.0, 2),
            "elo_prob_b": round(100.0 - p_elo * 100.0, 2),
            "elo_neutral_prob_a": round(p_neutral * 100.0, 2),
            "host_impact": round((p_elo - p_neutral) * 100.0, 2),
            "calibration_impact": round((p_cal - p_elo) * 100.0, 2),
            "form_impact": round((p_form - p_cal) * 100.0, 2),
            "style_impact": round((p_final - p_form) * 100.0, 2),
            "elo_a": round(float(sa["elo"]), 1),
            "elo_b": round(float(sb["elo"]), 1),
            "host_bonus_a": HOST_BONUS if host_diff > 0 else 0.0,
            "host_bonus_b": HOST_BONUS if host_diff < 0 else 0.0,
            "possession_a": sa["possession"],
            "possession_b": sb["possession"],
            "form_a": round(float(sa["form"]), 2),
            "form_b": round(float(sb["form"]), 2),
            "gf_a": round(float(sa["goals_scored"]), 2),
            "ga_a": round(float(sa["goals_conceded"]), 2),
            "gf_b": round(float(sb["goals_scored"]), 2),
            "ga_b": round(float(sb["goals_conceded"]), 2),
            "style_score_a": round(float(sa["style_score"]), 2),
            "style_score_b": round(float(sb["style_score"]), 2),
        }
        result["summary"] = self._summary(result, sa, sb)
        return result

    @staticmethod
    def _summary(r: dict, sa: Mapping, sb: Mapping) -> str:
        a, b = r["team_a"], r["team_b"]
        parts = []

        d_elo = r["elo_a"] - r["elo_b"] + r["host_bonus_a"] - r["host_bonus_b"]
        fav = a if d_elo >= 0 else b
        host_note = " (z bonusem gospodarza)" if (r["host_bonus_a"] or r["host_bonus_b"]) else ""
        parts.append(f"Elo: {fav} wyżej o {abs(d_elo):.0f} pkt{host_note} ({r['elo_prob_a']:.0f}% - {r['elo_prob_b']:.0f}%).")

        pa, pb = sa["possession"], sb["possession"]
        if not (pa != pa or pb != pb):  # brak NaN
            if abs(pa - pb) < 2.0:
                parts.append(f"Posiadanie piłki wyrównane ({pa:.0f}% vs {pb:.0f}%).")
            else:
                dom = a if pa > pb else b
                parts.append(f"Więcej piłki ma {dom} ({max(pa, pb):.0f}% vs {min(pa, pb):.0f}%).")

        gd_a = float(sa["goals_scored"]) - float(sa["goals_conceded"])
        gd_b = float(sb["goals_scored"]) - float(sb["goals_conceded"])
        form_edge = (float(sa["form"]) - float(sb["form"])) / FORM_SCALE + (gd_a - gd_b)
        if abs(form_edge) < 0.15:
            parts.append("Forma zbliżona.")
        else:
            better = a if form_edge > 0 else b
            sf, sg = (sa, gd_a) if better == a else (sb, gd_b)
            parts.append(
                f"Lepszą formę ma {better} ({float(sf['form']):+.1f} Elo/mecz, bilans bramek {sg:+.2f}/mecz)."
            )

        total = r["prob_a"] - r["elo_prob_a"]
        if abs(total) < 0.05:
            parts.append(f"Model hybrydowy nie zmienia oceny Elo ({r['prob_a']:.1f}% dla {a}).")
        else:
            parts.append(
                f"Model hybrydowy: {a} {r['prob_a']:.1f}% ({total:+.1f} pp vs Elo; forma {r['form_impact']:+.1f} pp, "
                f"styl {r['style_impact']:+.1f} pp)."
            )
        return " ".join(parts)

    # ---------------------------------------------------------- walidacja
    def _evaluate_holdout(self, history: pd.DataFrame) -> None:
        """Walidacja czasowa (trening < 2022, test >= 2022) na miękkim celu: log-loss."""
        train = history[history["date"].dt.year < HOLDOUT_YEAR]
        test = history[history["date"].dt.year >= HOLDOUT_YEAR]
        if len(train) == 0 or len(test) == 0:
            return
        models = self._fit_models(*self._training_set(train))

        elo_diff = (
            test["home_elo_pre"].to_numpy()
            - test["away_elo_pre"].to_numpy()
            + np.where(test["neutral"].to_numpy(), 0.0, HOME_ADVANTAGE)
        )
        X_full = self._features(
            elo_diff,
            test["home_form_pre"].to_numpy() - test["away_form_pre"].to_numpy(),
            test["home_gd_pre"].to_numpy() - test["away_gd_pre"].to_numpy(),
        )
        X_zero = X_full.copy()
        X_zero[:, 1:] = 0.0
        p_elo = _sigmoid(X_full[:, 0])

        hs, as_ = test["home_score"].to_numpy(), test["away_score"].to_numpy()
        s = np.where(hs > as_, 1.0, np.where(hs == as_, 0.5, 0.0))

        def logloss(p):
            p = np.clip(p, _EPS, 1 - _EPS)
            return float(np.mean(-(s * np.log(p) + (1 - s) * np.log(1 - p))))

        self.diagnostics = {
            "holdout_matches": int(len(test)),
            "logloss_elo": logloss(p_elo),
            "logloss_ensemble_no_form": logloss(self._blend(p_elo, X_zero, models)),
            "logloss_ensemble_with_form": logloss(self._blend(p_elo, X_full, models)),
        }
