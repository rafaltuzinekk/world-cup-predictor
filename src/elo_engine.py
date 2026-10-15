"""Silnik Elo (wzorowany na World Football Elo Ratings) liczony na meczach FIFA z DuckDB.

Główne cechy:
* bazowy rating 1500, domyślny start (rozgrzewka / warm-up) od 1980 roku,
* dynamiczne K zależne od rangi turnieju,
* mnożnik różnicy bramek G (standard World Football Elo),
* przewaga gospodarza +100 pkt (tylko przy oczekiwanym rezultacie, gdy ``neutral == False``),
* wskaźniki formy:
    - ``recent_form``          - średnia zmiana Elo z ostatnich 6 meczów (Krok 1),
    - ``form_rating_change``   - średnia zmiana Elo z ostatnich 8 meczów,
    - ``goals_scored_recent``  - średnia bramek strzelonych w ostatnich 8 meczach,
    - ``goals_conceded_recent``- średnia bramek straconych w ostatnich 8 meczach,
* opcjonalny ``cutoff_date`` (mecze ściśle PRZED tą datą) - "wehikuł czasu" bez wycieku danych.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date as date_type
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd

from .database import BASE_DIR

CURRENT_ELO_CSV = BASE_DIR / "data" / "processed" / "current_elo_ratings.csv"

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0
DEFAULT_START_YEAR = 1980  # okres warm-up: hierarchia między kontynentami zdąży się ukształtować
FORM_WINDOW = 6  # recent_form (zgodność z Krokiem 1)
FORM8_WINDOW = 8  # nowe metryki formy

K_FRIENDLY = 10
K_QUALIFIERS_AND_NATIONS_LEAGUE = 25
K_CONTINENTAL = 40
K_WORLD_CUP = 60
K_OTHER = 20

CONTINENTAL_FINALS = frozenset(
    {
        "UEFA Euro",
        "Copa América",
        "African Cup of Nations",
        "AFC Asian Cup",
        "Gold Cup",
    }
)

DateLike = Union[str, date_type, pd.Timestamp]


class EloEngine:
    def __init__(
        self,
        base_rating: float = BASE_RATING,
        home_advantage: float = HOME_ADVANTAGE,
        start_year: int = DEFAULT_START_YEAR,
        form_window: int = FORM_WINDOW,
        form8_window: int = FORM8_WINDOW,
    ) -> None:
        self.base_rating = base_rating
        self.home_advantage = home_advantage
        self.start_year = start_year
        self.form_window = form_window
        self.form8_window = form8_window

        self.ratings: Dict[str, float] = defaultdict(lambda: self.base_rating)
        self.matches_played: Dict[str, int] = defaultdict(int)
        self.last_match: Dict[str, pd.Timestamp] = {}
        self._recent_deltas: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.form_window)
        )
        # (zmiana Elo, bramki strzelone, bramki stracone) z ostatnich N meczów
        self._form8: Dict[str, Deque[Tuple[float, int, int]]] = defaultdict(
            lambda: deque(maxlen=self.form8_window)
        )
        self.history: List[dict] = []

    # ------------------------------------------------------------------ wzory
    @staticmethod
    def k_factor(tournament: str) -> int:
        """Waga meczu K zależna od rodzaju turnieju."""
        name = (tournament or "").strip()
        low = name.lower()
        if name == "Friendly":
            return K_FRIENDLY
        # eliminacje (każdego turnieju) oraz Ligi Narodów (UEFA / CONCACAF)
        if "qualification" in low or "nations league" in low:
            return K_QUALIFIERS_AND_NATIONS_LEAGUE
        if name == "FIFA World Cup":
            return K_WORLD_CUP
        if name in CONTINENTAL_FINALS:
            return K_CONTINENTAL
        return K_OTHER

    @staticmethod
    def goal_multiplier(goal_diff: int) -> float:
        """Mnożnik G: 0-1 gol -> 1.0, 2 gole -> 1.5, >=3 -> (11 + N) / 8."""
        n = abs(int(goal_diff))
        if n <= 1:
            return 1.0
        if n == 2:
            return 1.5
        return (11 + n) / 8.0

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """E_A = 1 / (1 + 10^((R_B - R_A) / 400))."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    # ------------------------------------------------------------------ forma
    def team_form(self, team: str) -> Dict[str, float]:
        """Wskaźniki formy z ostatnich ``form8_window`` meczów (zera, gdy brak meczów)."""
        window = self._form8.get(team)
        if not window:
            return {"form_rating_change": 0.0, "goals_scored_recent": 0.0, "goals_conceded_recent": 0.0}
        n = len(window)
        return {
            "form_rating_change": sum(w[0] for w in window) / n,
            "goals_scored_recent": sum(w[1] for w in window) / n,
            "goals_conceded_recent": sum(w[2] for w in window) / n,
        }

    def get_form_window(self, team: str) -> List[Tuple[float, int, int]]:
        """Kopia okna ostatnich meczów: [(delta_elo, gf, ga), ...] od najstarszego."""
        return list(self._form8.get(team, ()))

    def recent_form(self, team: str) -> float:
        """Średnia zmiana Elo w ostatnich ``form_window`` meczach (0.0 bez meczów)."""
        deltas = self._recent_deltas.get(team)
        return float(sum(deltas) / len(deltas)) if deltas else 0.0

    # ----------------------------------------------------------- aktualizacja
    def update_match(
        self,
        date: pd.Timestamp,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        tournament: str,
        neutral: bool,
    ) -> None:
        r_home, r_away = self.ratings[home], self.ratings[away]

        # Forma PRZED meczem (do trenowania modelu bez wycieku danych).
        fh, fa = self.team_form(home), self.team_form(away)

        # Bonus gospodarza tylko do obliczenia oczekiwanego rezultatu (nie zapisujemy go w ratingu).
        bonus = 0.0 if neutral else self.home_advantage
        e_home = self.expected_score(r_home + bonus, r_away)

        if home_score > away_score:
            w_home = 1.0
        elif home_score == away_score:
            w_home = 0.5
        else:
            w_home = 0.0

        k = self.k_factor(tournament)
        g = self.goal_multiplier(home_score - away_score)
        delta_home = k * g * (w_home - e_home)

        self.ratings[home] = r_home + delta_home
        self.ratings[away] = r_away - delta_home

        for team, delta, gf, ga in (
            (home, delta_home, home_score, away_score),
            (away, -delta_home, away_score, home_score),
        ):
            self.matches_played[team] += 1
            self.last_match[team] = date
            self._recent_deltas[team].append(delta)
            self._form8[team].append((delta, gf, ga))

        self.history.append(
            {
                "date": date,
                "home_team": home,
                "away_team": away,
                "tournament": tournament,
                "neutral": bool(neutral),
                "k": k,
                "g": g,
                "home_elo_pre": r_home,
                "away_elo_pre": r_away,
                "home_expected": e_home,
                "home_elo_delta": delta_home,
                "home_form_pre": fh["form_rating_change"],
                "away_form_pre": fa["form_rating_change"],
                "home_gd_pre": fh["goals_scored_recent"] - fh["goals_conceded_recent"],
                "away_gd_pre": fa["goals_scored_recent"] - fa["goals_conceded_recent"],
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    def fit(self, matches: pd.DataFrame, cutoff_date: Optional[DateLike] = None) -> "EloEngine":
        """Przetwarza mecze chronologicznie. Wymagane kolumny jak w ``raw_matches``.

        ``cutoff_date``: uwzględniane są tylko mecze ściśle wcześniejsze niż ta data.
        """
        played = matches.dropna(subset=["home_score", "away_score"]).copy()
        played["date"] = pd.to_datetime(played["date"])
        played = played[played["date"].dt.year >= self.start_year]
        if cutoff_date is not None:
            played = played[played["date"] < pd.Timestamp(cutoff_date)]
        sort_cols = ["date", "match_id"] if "match_id" in played.columns else ["date"]
        played = played.sort_values(sort_cols, kind="stable")

        for row in played.itertuples(index=False):
            self.update_match(
                date=row.date,
                home=row.home_team,
                away=row.away_team,
                home_score=int(row.home_score),
                away_score=int(row.away_score),
                tournament=row.tournament,
                neutral=bool(row.neutral),
            )
        return self

    def fit_from_db(
        self, con: duckdb.DuckDBPyConnection, cutoff_date: Optional[DateLike] = None
    ) -> "EloEngine":
        matches = con.execute(
            """
            SELECT match_id, date, home_team, away_team, home_score, away_score,
                   tournament, neutral
            FROM raw_matches
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
              AND year(date) >= ?
            ORDER BY date, match_id
            """,
            [self.start_year],
        ).df()
        return self.fit(matches, cutoff_date=cutoff_date)

    # ---------------------------------------------------------------- wyniki
    def ratings_table(self) -> pd.DataFrame:
        rows = []
        for team, rating in self.ratings.items():
            if self.matches_played[team] <= 0:
                continue
            form = self.team_form(team)
            rows.append(
                {
                    "team": team,
                    "current_elo": rating,
                    "matches_played": self.matches_played[team],
                    "recent_form": self.recent_form(team),
                    "form_rating_change": form["form_rating_change"],
                    "goals_scored_recent": form["goals_scored_recent"],
                    "goals_conceded_recent": form["goals_conceded_recent"],
                    "last_match_date": self.last_match[team].date(),
                }
            )
        df = pd.DataFrame(rows)
        df = df.sort_values("current_elo", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", df.index + 1)
        return df

    def history_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def save(
        self,
        con: duckdb.DuckDBPyConnection,
        csv_path: Optional[Path] = CURRENT_ELO_CSV,
        table_name: str = "elo_ratings",
        history_table_name: Optional[str] = "elo_history",
    ) -> pd.DataFrame:
        """Zapisuje ranking do tabeli ``elo_ratings`` (+ ``elo_history``) i do CSV (kompatybilność wsteczna)."""
        ratings = self.ratings_table()

        con.register("ratings_df", ratings)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM ratings_df")
        con.unregister("ratings_df")

        if history_table_name:
            history = self.history_table()
            con.register("history_df", history)
            con.execute(f"CREATE OR REPLACE TABLE {history_table_name} AS SELECT * FROM history_df")
            con.unregister("history_df")

        if csv_path is not None:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            # Kolumny `team` i `current_elo` pozostają pierwsze - tak czyta je app.py.
            cols = [
                "team",
                "current_elo",
                "matches_played",
                "recent_form",
                "form_rating_change",
                "goals_scored_recent",
                "goals_conceded_recent",
                "rank",
            ]
            ratings[cols].to_csv(csv_path, index=False)
        return ratings
