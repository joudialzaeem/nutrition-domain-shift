from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "image_diagnostics" / "foodbd_test180_verified_image_features_and_errors.csv")

with open(DATA / "target_scaler.json", encoding="utf-8") as f:
    scaler = json.load(f)

targets = scaler["targets"]
mean = np.asarray(scaler["mean"], dtype=float)
std = np.asarray(scaler["std"], dtype=float)

true_values = df[targets].to_numpy(dtype=float)
standardized_distance = np.abs((true_values - mean) / std)
df["nutritional_distance"] = np.sqrt((standardized_distance ** 2).sum(axis=1))

error_columns = [f"avg_{target}_ae" for target in targets]
errors = df[error_columns].to_numpy(dtype=float)
df["aggregate_standardized_error"] = (errors / std).sum(axis=1)

rho_all, p_all = spearmanr(
    df["nutritional_distance"],
    df["aggregate_standardized_error"]
)

carb_distance = standardized_distance[:, targets.index("carbs")]
rho_carbs, p_carbs = spearmanr(carb_distance, df["avg_carbs_ae"])

ordered = df.sort_values("nutritional_distance")
closest = ordered.head(len(ordered) // 4)
furthest = ordered.tail(len(ordered) // 4)

summary = pd.DataFrame([
    {
        "comparison": "overall_nutritional_distance_vs_aggregate_standardized_error",
        "rho": rho_all,
        "p_value": p_all
    },
    {
        "comparison": "carbohydrate_distance_vs_carbohydrate_error",
        "rho": rho_carbs,
        "p_value": p_carbs
    }
])

quartiles = pd.DataFrame([
    {
        "group": "closest_25_percent",
        "n": len(closest),
        "mean_calorie_error": closest["avg_calories_ae"].mean(),
        "mean_carbohydrate_error": closest["avg_carbs_ae"].mean()
    },
    {
        "group": "furthest_25_percent",
        "n": len(furthest),
        "mean_calorie_error": furthest["avg_calories_ae"].mean(),
        "mean_carbohydrate_error": furthest["avg_carbs_ae"].mean()
    }
])

RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "foodbd_nutritional_shift_per_image.csv", index=False)
summary.to_csv(RESULTS / "foodbd_nutritional_shift_correlations.csv", index=False)
quartiles.to_csv(RESULTS / "foodbd_nutritional_shift_quartiles.csv", index=False)

print(summary.to_string(index=False))
print()
print(quartiles.to_string(index=False))
