"""Script 23 — normalize the SBR 10Y game-odds archive into odds_games.parquet.

Source: artifacts/odds_raw/nhl_archive_10Y.json (game-level, seasons 2011-2021,
opening + closing moneylines and totals, 99.9% closing coverage).
Normalizations: nickname -> NHL triCode, int date -> date, American odds ->
implied probability + de-vigged closing probs (2-way proportional).

Output: artifacts/odds_games.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(r"D:\optbot\artifacts")

NICK2TRI = {
    "Bruins": "BOS", "Sabres": "BUF", "Redwings": "DET", "Red Wings": "DET",
    "Panthers": "FLA", "Canadiens": "MTL", "Senators": "OTT", "Lightning": "TBL",
    "Mapleleafs": "TOR", "Maple Leafs": "TOR", "Hurricanes": "CAR",
    "Bluejackets": "CBJ", "Blue Jackets": "CBJ", "Devils": "NJD",
    "Islanders": "NYI", "Rangers": "NYR", "Flyers": "PHI", "Penguins": "PIT",
    "Capitals": "WSH", "Blackhawks": "CHI", "Avalanche": "COL", "Stars": "DAL",
    "Wild": "MIN", "Predators": "NSH", "Blues": "STL", "Jets": "WPG",
    "Ducks": "ANA", "Coyotes": "ARI", "Flames": "CGY", "Oilers": "EDM",
    "Kings": "LAK", "Sharks": "SJS", "Canucks": "VAN", "Goldenknights": "VGK",
    "Golden Knights": "VGK", "Kraken": "SEA", "Thrashers": "ATL",
    "St.Louis": "STL", "St. Louis": "STL", "WinnipegJets": "WPG",
    "Phoenix": "ARI", "Arizonas": "ARI", "Arizona": "ARI",
    "Tampa": "TBL", "Tampa Bay": "TBL", "NY Islanders": "NYI",
    "NY Rangers": "NYR", "SeattleKraken": "SEA",
}


def american_to_prob(o):
    o = pd.to_numeric(o, errors="coerce")
    return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))


if __name__ == "__main__":
    d = json.load(open(ART / "odds_raw" / "nhl_archive_10Y.json"))
    df = pd.DataFrame(d)
    df["date"] = pd.to_datetime(df.date.astype(int).astype(str), format="%Y%m%d")
    for side in ("home", "away"):
        df[f"{side}_tri"] = df[f"{side}_team"].map(NICK2TRI)
        for ph in ("open", "close"):
            df[f"{side}_{ph}_p_raw"] = american_to_prob(df[f"{side}_{ph}_ml"])
    unmapped = df[df.home_tri.isna()].home_team.unique().tolist()
    if unmapped:
        n_bad = int(df.home_tri.isna().sum() + df.away_tri.isna().sum())
        print(f"UNMAPPED nicknames (dropping {n_bad} rows): {unmapped}")
        df = df[df.home_tri.notna() & df.away_tri.notna()].copy()
    for c in ("home_final", "away_final"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # de-vig closing (proportional): p_home + p_away scaled to 1
    s = df.home_close_p_raw + df.away_close_p_raw
    df["home_close_p_fair"] = df.home_close_p_raw / s
    df["away_close_p_fair"] = df.away_close_p_raw / s
    df["overround_close"] = s - 1.0
    keep = ["season", "date", "home_tri", "away_tri", "home_final", "away_final",
            "home_open_ml", "away_open_ml", "home_close_ml", "away_close_ml",
            "open_over_under", "close_over_under", "close_over_under_odds",
            "home_close_p_fair", "away_close_p_fair", "overround_close",
            "home_open_p_raw", "away_open_p_raw"]
    out = df[keep]
    out.to_parquet(ART / "odds_games.parquet", index=False)
    print(f"wrote odds_games.parquet: {len(out):,} games, "
          f"seasons {out.season.min()}-{out.season.max()}, "
          f"median overround {out.overround_close.median():.3f}")
