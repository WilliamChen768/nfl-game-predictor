from pathlib import Path

# Project root folder (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

FIRST_SEASON = 2006

# Teams that relocated: old code -> current code
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}