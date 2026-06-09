#!/usr/bin/env python3
"""Refresh UFC fighter DB from latest Kaggle dataset.

Downloads the jossilva3110/ufc-dataset-1994-2026 dataset and updates
fighter_lookup.json with current per-fighter stats (wins, losses,
significant strikes, takedowns, physicals, etc.).

Usage:
    python src/scripts/refresh_ufc_fighters.py          # dry run (show changes)
    python src/scripts/refresh_ufc_fighters.py --apply  # write updated DB
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

MODEL_DIR = Path("models/ufc")
FIGHTER_DB = MODEL_DIR / "fighter_lookup.json"


def download_kaggle_fighters() -> pd.DataFrame:
    """Download latest fighters CSV from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        print("Install kagglehub: pip install kagglehub")
        sys.exit(1)

    path = kagglehub.dataset_download("jossilva3110/ufc-dataset-1994-2026")
    csv_path = Path(path) / "ufc_fighters_final.csv"
    if not csv_path.exists():
        print(f"Fighters CSV not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(str(csv_path))
    print(f"  Downloaded {len(df)} fighters from Kaggle")
    return df


def parse_height_cm(height_str: str) -> float:
    """Convert '5\' 11\"' or '180 cm' to cm."""
    import re
    s = str(height_str).strip().strip('"').strip("'")
    # "5' 11\""
    m = re.match(r"(\d+)'?\s*(\d+)?", s)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2)) if m.group(2) else 0
        return round((feet * 12 + inches) * 2.54, 1)
    # "180 cm" or just "180"
    m = re.match(r"(\d+\.?\d*)", s)
    if m:
        val = float(m.group(1))
        return val if val > 50 else val  # already cm
    return 178.0


def parse_weight_lbs(weight_str: str) -> float:
    """Convert weight string to lbs."""
    import re
    s = str(weight_str).strip().lower().replace(" lbs", "").replace(" lb", "")
    m = re.match(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return 170.0


def parse_reach_cm(reach_str: str) -> float:
    """Convert reach string to cm."""
    import re
    s = str(reach_str).strip().strip('"')
    # "72\"" or "72"
    m = re.match(r"(\d+\.?\d*)", s)
    if m:
        val = float(m.group(1))
        return round(val * 2.54, 1) if val < 50 else val  # inches -> cm
    return 183.0


def refresh_fighter_db(apply: bool = False):
    """Download and merge updated fighter stats."""
    kaggle = download_kaggle_fighters()

    # Load existing DB
    if FIGHTER_DB.exists():
        with open(FIGHTER_DB) as f:
            db = json.load(f)
    else:
        db = {}
    print(f"  Existing DB: {len(db)} fighters")

    # Build a lookup key -> row mapping from Kaggle
    kaggle_lookup = {}
    for _, row in kaggle.iterrows():
        name = str(row.get("Fighter_Name", "")).strip()
        if not name or name == "nan":
            continue
        # Normalize: strip middle initials, handle common variants
        key = name.lower()
        kaggle_lookup[key] = row

    new_count = 0
    updated_count = 0
    stale_fixes = []

    for name in list(db.keys()):
        key = name.lower()
        if key in kaggle_lookup:
            kr = kaggle_lookup[key]
            kg_wins = int(kr.get("Wins", 0) or 0)
            kg_losses = int(kr.get("Losses", 0) or 0)
            old_wins = db[name].get("wins", 0)
            old_losses = db[name].get("losses", 0)

            # Check if record is stale (different from Kaggle)
            if old_wins != kg_wins or old_losses != kg_losses:
                db[name]["wins"] = kg_wins
                db[name]["losses"] = kg_losses
                db[name]["height_cms"] = parse_height_cm(kr.get("Height", ""))
                db[name]["reach_cms"] = parse_reach_cm(kr.get("Reach", ""))
                db[name]["weight_lbs"] = parse_weight_lbs(kr.get("Weight", ""))
                db[name]["avg_sig_str_landed"] = float(kr.get("SLpM", 27.0) or 27.0)
                db[name]["avg_sig_str_pct"] = float(str(kr.get("Str_Acc", "0.48")).replace("%", "")) / 100.0
                db[name]["avg_td_landed"] = float(str(kr.get("TD_Avg", "1.3")).replace("%", ""))
                db[name]["avg_td_pct"] = float(str(kr.get("TD_Acc", "0.35")).replace("%", "")) / 100.0
                db[name]["avg_sub_att"] = float(str(kr.get("Sub_Avg", "0.5")).replace("%", ""))
                db[name]["stance"] = str(kr.get("Stance", "Orthodox")).strip().lower()
                updated_count += 1
                stale_fixes.append(
                    f"    {name}: {old_wins}-{old_losses} → {kg_wins}-{kg_losses}"
                )

    # Add new fighters not in DB
    for key, kr in kaggle_lookup.items():
        name = str(kr.get("Fighter_Name", "")).strip()
        if not name:
            continue
        if name not in db:
            db[name] = {
                "avg_sig_str_landed": float(kr.get("SLpM", 27.0) or 27.0),
                "avg_sig_str_pct": (float(str(kr.get("Str_Acc", "0.48")).replace("%", "")) / 100.0),
                "avg_td_landed": float(str(kr.get("TD_Avg", "1.3")).replace("%", "")),
                "avg_td_pct": (float(str(kr.get("TD_Acc", "0.35")).replace("%", "")) / 100.0),
                "avg_sub_att": float(str(kr.get("Sub_Avg", "0.5")).replace("%", "")),
                "wins": int(kr.get("Wins", 0) or 0),
                "losses": int(kr.get("Losses", 0) or 0),
                "total_rounds_fought": 10,
                "height_cms": parse_height_cm(kr.get("Height", "")),
                "reach_cms": parse_reach_cm(kr.get("Reach", "")),
                "weight_lbs": parse_weight_lbs(kr.get("Weight", "")),
                "age": 30,
                "odds": 0,
                "win_by_ko_tko": 0,
                "win_by_submission": 0,
                "win_by_decision_unanimous": 0,
                "win_by_decision_split": 0,
                "win_by_decision_majority": 0,
                "win_by_tko_doctor_stoppage": 0,
                "current_win_streak": 1,
                "current_lose_streak": 0,
                "longest_win_streak": 3,
                "total_title_bouts": 0,
                "match_weightclass_rank": 50,
                "stance": str(kr.get("Stance", "Orthodox")).strip().lower(),
                "weight_class": "middleweight",
                "avg_fight_time": 652,
                "total_fight_time_secs": 652,
            }
            new_count += 1

    print(f"\n  Summary:")
    print(f"    Updated: {updated_count} fighters (stale records fixed)")
    print(f"    New:     {new_count} fighters added from Kaggle")
    print(f"    Total:   {len(db)} fighters")

    if stale_fixes:
        print(f"\n  Notable record fixes:")
        for fix in stale_fixes[:20]:
            print(fix)
        if len(stale_fixes) > 20:
            print(f"    ... and {len(stale_fixes) - 20} more")

    if apply:
        # Backup
        bak = MODEL_DIR / "fighter_lookup.json.bak"
        if FIGHTER_DB.exists():
            bak.write_text(FIGHTER_DB.read_text())

        FIGHTER_DB.write_text(json.dumps(db, indent=2))
        print(f"\n  ✅ Saved {len(db)} fighters to {FIGHTER_DB}")
        print(f"     Backup: {bak}")
    else:
        print(f"\n  DRY RUN — use --apply to write changes")
        print(f"     Target: {FIGHTER_DB}")


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    refresh_fighter_db(apply=apply)
