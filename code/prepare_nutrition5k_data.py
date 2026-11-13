from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NUTRITION5K = DATA / "nutrition5k"

source = pd.read_csv(NUTRITION5K / "nutrition5k_final_metadata.csv")

ingredient_records = []
with open(NUTRITION5K / "dish_metadata_cafe1.csv", encoding="utf-8", newline="") as f:
    for row in csv.reader(f):
        names = []
        for i in range(0, len(row[6:]), 7):
            names.append(row[6:][i + 1].strip().lower())
        ingredient_records.append({
            "dish_id": row[0],
            "has_plate_only": any(name == "plate only" for name in names),
            "has_deprecated_ingredient": any("deprecated" in name for name in names),
            "ingredient_names": " | ".join(names)
        })

ingredient_flags = pd.DataFrame(ingredient_records)
data = source.merge(ingredient_flags, on="dish_id", how="left")

data["known_phone_issue"] = data["dish_id"] == "dish_1559172356"
data["known_extreme_mass_issue"] = data["dish_id"] == "dish_1560974769"
data["invalid_numeric_target"] = (data["calories"] <= 0) | (data["mass"] <= 0)

data["quality_problem"] = (
    data["has_plate_only"]
    | data["has_deprecated_ingredient"]
    | data["known_phone_issue"]
    | data["known_extreme_mass_issue"]
    | data["invalid_numeric_target"]
)

def exclusion_reason(row):
    reasons = []
    if row["has_plate_only"]:
        reasons.append("plate_only")
    if row["has_deprecated_ingredient"]:
        reasons.append("deprecated_ingredient")
    if row["known_phone_issue"]:
        reasons.append("non_food_phone_image")
    if row["known_extreme_mass_issue"]:
        reasons.append("documented_extreme_mass_error")
    if row["invalid_numeric_target"] and not row["has_plate_only"]:
        reasons.append("invalid_numeric_target")
    return "; ".join(reasons)

data["exclusion_reason"] = data.apply(exclusion_reason, axis=1)

excluded = data[data["quality_problem"]].reset_index(drop=True)
clean = data[~data["quality_problem"]].reset_index(drop=True)

assert len(clean) == 3227
assert len(excluded) == 35
assert clean["dish_id"].is_unique

clean.to_csv(NUTRITION5K / "nutrition5k_clean_metadata.csv", index=False)
excluded.to_csv(NUTRITION5K / "nutrition5k_excluded_dishes.csv", index=False)

model_data = clean.rename(columns={"split": "official_split"}).copy()
model_data["dish_timestamp"] = model_data["dish_id"].str.extract(r"(\d+)$")[0].astype("int64")
model_data["collection_datetime"] = (
    pd.to_datetime(model_data["dish_timestamp"], unit="s", utc=True)
    .dt.tz_convert("America/Los_Angeles")
)
model_data["collection_date"] = model_data["collection_datetime"].dt.date.astype(str)

training_pool = model_data[model_data["official_split"] == "train"].copy()
official_test = model_data[model_data["official_split"] == "test"].copy()

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(
    splitter.split(training_pool, groups=training_pool["collection_date"])
)

train = training_pool.iloc[train_idx].copy()
val = training_pool.iloc[val_idx].copy()
train["model_split"] = "train"
val["model_split"] = "val"
official_test["model_split"] = "test"

model_splits = pd.concat([train, val, official_test], ignore_index=True)

counts = model_splits["model_split"].value_counts().to_dict()
assert counts == {"train": 2245, "test": 498, "val": 484}
assert not set(train["collection_date"]) & set(val["collection_date"])
assert not set(train["dish_id"]) & set(val["dish_id"])
assert not set(train["dish_id"]) & set(official_test["dish_id"])
assert not set(val["dish_id"]) & set(official_test["dish_id"])

model_splits.to_csv(DATA / "nutrition5k_model_splits.csv", index=False)

targets = ["calories", "protein", "carbs", "fat"]
train_targets = train[targets].to_numpy(dtype=float)
scaler = {
    "targets": targets,
    "mean": train_targets.mean(axis=0).tolist(),
    "std": train_targets.std(axis=0).tolist(),
    "fitted_on": "Nutrition5k training split only"
}

with open(DATA / "target_scaler.json", "w", encoding="utf-8") as f:
    json.dump(scaler, f, indent=4)

print("clean:", len(clean))
print("excluded:", len(excluded))
print("split counts:", counts)
print("target mean:", scaler["mean"])
print("target std:", scaler["std"])
