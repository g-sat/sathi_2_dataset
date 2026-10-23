"""Write one patient's full food timeline to a CSV you can open in Excel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.foods import ACTION_NAMES, DIET_PATTERN_NAMES, FOOD_CATALOG, REGION_NAMES

MEAL_NAMES = ("breakfast", "lunch", "snack", "dinner")
DATA_ROOT = ROOT / "data" / "synthetic"


def _find_row(sequence_id: int, split: str | None) -> pd.Series:
    splits = (split,) if split is not None else ("train", "validation", "test")
    for name in splits:
        folder = DATA_ROOT / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.parquet")):
            frame = pd.read_parquet(path)
            matched = frame.loc[frame["sequence_id"] == sequence_id]
            if not matched.empty:
                return matched.iloc[0]
    searched = split if split is not None else "train, validation, and test"
    raise SystemExit(f"patient {sequence_id} is not in {searched} under {DATA_ROOT}")


def _timeline(row: pd.Series) -> pd.DataFrame:
    food_ids = [int(item) for item in row.food_ids]
    names = [
        FOOD_CATALOG.names[item] if 0 <= item < len(FOOD_CATALOG.names) else f"unknown_{item}"
        for item in food_ids
    ]
    cuisine_ids = [int(item) for item in row.cuisine]
    return pd.DataFrame(
        {
            "day": [int(item) for item in row.day_index],
            "meal": [MEAL_NAMES[index % len(MEAL_NAMES)] for index in range(len(food_ids))],
            "hour": [round(float(item), 2) for item in row.hours],
            "food": names,
            "cuisine": ["south_asian" if item == 0 else "western" for item in cuisine_ids],
            "carbs_g": [round(float(item), 2) for item in row.carbs_g],
            "protein_g": [round(float(item), 2) for item in row.protein_g],
            "fat_g": [round(float(item), 2) for item in row.fat_g],
            "fiber_g": [round(float(item), 2) for item in row.fiber_g],
            "added_sugar_g": [round(float(item), 2) for item in row.added_sugar_g],
            "gi": [round(float(item), 1) for item in row.gi],
            "walk_minutes": [round(float(item), 1) for item in row.walk_minutes],
            "observed": [bool(item) for item in row.observed],
            "glucose_2h": [round(float(item), 2) for item in row.glucose_2h],
            "glucose_delta": [round(float(item), 2) for item in row.glucose_delta],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one simulated patient's meal timeline.")
    parser.add_argument("--id", type=int, default=0, help="sequence_id of the patient")
    parser.add_argument("--split", choices=("train", "validation", "test"), default=None)
    parser.add_argument("--head", type=int, default=12, help="rows to print in the terminal")
    args = parser.parse_args()

    row = _find_row(args.id, args.split)
    timeline = _timeline(row)
    destination = DATA_ROOT / "views" / f"patient_{args.id}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    timeline.to_csv(destination, index=False)

    pattern_id = int(row.diet_pattern)
    region_id = int(row.region_id)
    action_id = int(row.y_class)
    pattern = DIET_PATTERN_NAMES[pattern_id] if 0 <= pattern_id < len(DIET_PATTERN_NAMES) else str(pattern_id)
    region = REGION_NAMES[region_id] if 0 <= region_id < len(REGION_NAMES) else str(region_id)
    action = ACTION_NAMES[action_id] if 0 <= action_id < len(ACTION_NAMES) else str(action_id)
    print(f"patient {int(row.sequence_id)}")
    print(f"pattern {pattern}")
    print(f"region {region}")
    print(
        f"age {float(row.age_years):.1f}  sex {int(row.sex)}  "
        f"bmi {float(row.bmi):.1f}  family_history {int(row.family_history)}"
    )
    print(
        f"fasting {float(row.fasting_glucose):.1f} mg/dL  "
        f"mean 2-hour {float(row.y_value):.1f} mg/dL  action {action}"
    )
    print(f"meals {int(row.length)}  days {int(timeline['day'].iloc[-1]) + 1}")
    print(timeline.head(args.head).to_string(index=False))
    print(f"full timeline: {destination}")


if __name__ == "__main__":
    main()
