"""Warstwa danych: DuckDB + filtr oficjalnych członków FIFA.

Nazwy drużyn są ustandaryzowane do konwencji ``data/raw/results.csv``
(np. "United States", "South Korea", "Ivory Coast", "Czech Republic").
Aliasy z innych źródeł (np. FBref: "Czechia", "Korea Republic") są mapowane
na tę konwencję w ``TEAM_ALIASES``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "world_cup.duckdb"
RESULTS_CSV = BASE_DIR / "data" / "raw" / "results.csv"
FBREF_CSV = BASE_DIR / "data" / "processed" / "fbref_group_stats.csv"

# --------------------------------------------------------------------------
# 211 oficjalnych członków FIFA (nazwy jak w results.csv), wg konfederacji
# --------------------------------------------------------------------------
FIFA_MEMBERS_BY_CONFEDERATION: Dict[str, List[str]] = {
    "AFC": [  # 46
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei",
        "Cambodia", "China", "Guam", "Hong Kong", "India", "Indonesia", "Iran",
        "Iraq", "Japan", "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon",
        "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal",
        "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
        "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria",
        "Taiwan", "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan",
        "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
    ],
    "CAF": [  # 54
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
        "Congo", "DR Congo", "Djibouti", "Egypt", "Equatorial Guinea",
        "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
        "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya",
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco",
        "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda",
        "São Tomé and Príncipe", "Senegal", "Seychelles", "Sierra Leone",
        "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo",
        "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    ],
    "CONCACAF": [  # 35
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
        "Belize", "Bermuda", "British Virgin Islands", "Canada",
        "Cayman Islands", "Costa Rica", "Cuba", "Curaçao", "Dominica",
        "Dominican Republic", "El Salvador", "Grenada", "Guatemala", "Guyana",
        "Haiti", "Honduras", "Jamaica", "Mexico", "Montserrat", "Nicaragua",
        "Panama", "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Suriname", "Trinidad and Tobago",
        "Turks and Caicos Islands", "United States",
        "United States Virgin Islands",
    ],
    "CONMEBOL": [  # 10
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ],
    "OFC": [  # 11
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia",
        "New Zealand", "Papua New Guinea", "Samoa", "Solomon Islands", "Tahiti",
        "Tonga", "Vanuatu",
    ],
    "UEFA": [  # 55
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
        "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
        "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo",
        "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Montenegro", "Netherlands", "North Macedonia",
        "Northern Ireland", "Norway", "Poland", "Portugal", "Republic of Ireland",
        "Romania", "Russia", "San Marino", "Scotland", "Serbia", "Slovakia",
        "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine",
        "Wales",
    ],
}

FIFA_MEMBERS: List[str] = sorted(
    team for teams in FIFA_MEMBERS_BY_CONFEDERATION.values() for team in teams
)
assert len(FIFA_MEMBERS) == len(set(FIFA_MEMBERS)) == 211, (
    f"Lista FIFA powinna mieć 211 unikalnych drużyn, jest {len(set(FIFA_MEMBERS))}"
)

# Aliasy nazw z innych źródeł -> konwencja results.csv
TEAM_ALIASES: Dict[str, str] = {
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Bosnia–Herz": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "USA": "United States",
    "China PR": "China",
    "Ireland": "Republic of Ireland",
    "Swaziland": "Eswatini",
    "Macao": "Macau",
    "Curacao": "Curaçao",
    "Sao Tome and Principe": "São Tomé and Príncipe",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Timor Leste": "Timor-Leste",
}


def normalize_team_name(name: str) -> str:
    """Zwraca nazwę drużyny w konwencji results.csv."""
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def get_connection(db_path: Path = DB_PATH, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def _sql_path(path: Path) -> str:
    """Ścieżka w postaci bezpiecznej dla literału SQL (slashe, escape apostrofów)."""
    return Path(path).as_posix().replace("'", "''")


def init_database(
    db_path: Path = DB_PATH,
    results_csv: Path = RESULTS_CSV,
    fbref_csv: Path = FBREF_CSV,
) -> duckdb.DuckDBPyConnection:
    """Tworzy (od zera) tabele ``fifa_members``, ``raw_matches`` i ``team_stats_fbref``.

    ``raw_matches`` zawiera wyłącznie mecze, w których OBIE drużyny są
    oficjalnymi członkami FIFA (filtr realizowany w SQL). Mecze bez wyniku
    (przyszłe terminarze) zostają w tabeli - silnik Elo ignoruje je sam.
    """
    con = get_connection(db_path)

    # --- fifa_members -------------------------------------------------------
    members_df = pd.DataFrame(
        [(team, conf) for conf, teams in FIFA_MEMBERS_BY_CONFEDERATION.items() for team in teams],
        columns=["team", "confederation"],
    )
    con.register("members_df", members_df)
    con.execute("CREATE OR REPLACE TABLE fifa_members AS SELECT * FROM members_df")
    con.unregister("members_df")

    # --- raw_matches (filtr FIFA w SQL) --------------------------------------
    con.execute(
        f"""
        CREATE OR REPLACE TABLE raw_matches AS
        WITH src AS (
            SELECT
                row_number() OVER () AS match_id,
                CAST(date AS DATE)          AS date,
                TRIM(home_team)             AS home_team,
                TRIM(away_team)             AS away_team,
                CAST(home_score AS INTEGER) AS home_score,
                CAST(away_score AS INTEGER) AS away_score,
                tournament,
                city,
                country,
                upper(neutral) = 'TRUE'     AS neutral
            FROM read_csv(
                '{_sql_path(results_csv)}',
                header = true,
                nullstr = ['NA', ''],
                types = {{'home_score': 'VARCHAR', 'away_score': 'VARCHAR', 'neutral': 'VARCHAR'}}
            )
        )
        SELECT s.*
        FROM src s
        WHERE s.home_team IN (SELECT team FROM fifa_members)
          AND s.away_team IN (SELECT team FROM fifa_members)
        ORDER BY s.date, s.match_id
        """
    )

    # --- team_stats_fbref ----------------------------------------------------
    fbref_df = pd.read_csv(fbref_csv)
    fbref_df.insert(0, "team_fbref", fbref_df["team"])
    fbref_df["team"] = fbref_df["team"].map(normalize_team_name)
    con.register("fbref_df", fbref_df)
    con.execute("CREATE OR REPLACE TABLE team_stats_fbref AS SELECT * FROM fbref_df")
    con.unregister("fbref_df")

    return con


def data_quality_report(con: duckdb.DuckDBPyConnection) -> Dict[str, object]:
    """Krótkie podsumowanie jakości danych po filtrowaniu."""
    raw_total = con.execute(
        "SELECT count(*) FROM read_csv(?, header = true, nullstr = ['NA', ''], all_varchar = true)",
        [RESULTS_CSV.as_posix()],
    ).fetchone()[0]
    kept = con.execute("SELECT count(*) FROM raw_matches").fetchone()[0]
    teams_in_matches = con.execute(
        "SELECT count(DISTINCT t) FROM (SELECT home_team t FROM raw_matches "
        "UNION SELECT away_team FROM raw_matches)"
    ).fetchone()[0]
    missing_members = [
        r[0]
        for r in con.execute(
            "SELECT team FROM fifa_members WHERE team NOT IN "
            "(SELECT home_team FROM raw_matches UNION SELECT away_team FROM raw_matches) "
            "ORDER BY team"
        ).fetchall()
    ]
    fbref_unmatched = [
        r[0]
        for r in con.execute(
            "SELECT team FROM team_stats_fbref WHERE team NOT IN (SELECT team FROM fifa_members)"
        ).fetchall()
    ]
    return {
        "fifa_members": len(FIFA_MEMBERS),
        "raw_rows_in_csv": raw_total,
        "matches_kept": kept,
        "matches_dropped": raw_total - kept,
        "teams_in_matches": teams_in_matches,
        "fifa_members_without_matches": missing_members,
        "fbref_teams_not_in_fifa": fbref_unmatched,
    }


if __name__ == "__main__":
    connection = init_database()
    print(data_quality_report(connection))
    connection.close()
