from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "five_meal"

predictions = pd.read_csv(
    RESULTS / "five_meal_predictions.csv"
)

labels = pd.read_csv(
    RESULTS / "five_meal_label_ground_truth.csv"
)

truth = labels[
    [
        "meal_id",
        "calories_kcal",
        "protein_g",
        "carbs_g_min",
        "carbs_g_max_exclusive",
        "fat_g",
    ]
].copy()

truth["true_carbs"] = (
    truth["carbs_g_min"]
    + truth["carbs_g_max_exclusive"]
) / 2.0

truth = truth.rename(columns={
    "calories_kcal": "true_calories",
    "protein_g": "true_protein",
    "fat_g": "true_fat",
})

truth = truth[
    [
        "meal_id",
        "true_calories",
        "true_protein",
        "true_carbs",
        "true_fat",
    ]
]

data = predictions.merge(
    truth,
    on="meal_id",
    how="left",
    validate="many_to_one",
)

for target in ["calories", "protein", "carbs", "fat"]:
    data[f"ae_{target}"] = (
        data[f"pred_{target}"]
        - data[f"true_{target}"]
    ).abs()

data.to_csv(
    RESULTS / "five_meal_predictions_with_errors.csv",
    index=False,
)

baseline = data[
    data["condition_id"] == 1
][
    [
        "meal_id",
        "model",
        "variant",
        "pred_calories",
        "pred_protein",
        "pred_carbs",
        "pred_fat",
    ]
].copy()

baseline = baseline.rename(columns={
    "pred_calories": "base_pred_calories",
    "pred_protein": "base_pred_protein",
    "pred_carbs": "base_pred_carbs",
    "pred_fat": "base_pred_fat",
})

robustness = data.merge(
    baseline,
    on=["meal_id", "model", "variant"],
    how="left",
    validate="many_to_one",
)

for target in ["calories", "protein", "carbs", "fat"]:
    robustness[f"dev_{target}"] = (
        robustness[f"pred_{target}"]
        - robustness[f"base_pred_{target}"]
    ).abs()

robustness.to_csv(
    RESULTS / "five_meal_robustness_deviation.csv",
    index=False,
)

condition_summary = (
    robustness.groupby(
        ["condition_id", "condition"],
        as_index=False,
    )[
        [
            "dev_calories",
            "dev_protein",
            "dev_carbs",
            "dev_fat",
        ]
    ]
    .mean()
)

labels_map = {
    1: "Baseline",
    2: "Dim light",
    3: "Side angle",
    4: "Partial crop",
    5: "Mixed",
}

condition_summary["label"] = (
    condition_summary["condition_id"]
    .map(labels_map)
)

condition_summary.to_csv(
    RESULTS / "five_meal_condition_deviation_summary.csv",
    index=False,
)

mae_by_variant = (
    data.groupby("variant", as_index=False)[
        [
            "ae_calories",
            "ae_protein",
            "ae_carbs",
            "ae_fat",
        ]
    ]
    .mean()
)

variant_order = pd.Categorical(
    mae_by_variant["variant"],
    categories=["source", "mmd"],
    ordered=True,
)

mae_by_variant = (
    mae_by_variant.assign(_order=variant_order)
    .sort_values("_order")
    .drop(columns="_order")
    .reset_index(drop=True)
)

mae_by_variant.to_csv(
    RESULTS / "five_meal_mae_by_variant.csv",
    index=False,
)

mae_by_condition = (
    data.groupby(
        ["condition_id", "condition"],
        as_index=False,
    )[
        [
            "ae_calories",
            "ae_protein",
            "ae_carbs",
            "ae_fat",
        ]
    ]
    .mean()
)

mae_by_condition.to_csv(
    RESULTS / "five_meal_mae_by_condition.csv",
    index=False,
)

calorie_by_model_variant = (
    data.groupby(
        ["model", "variant"],
        as_index=False,
    )["ae_calories"]
    .mean()
)

calorie_by_model_variant.to_csv(
    RESULTS / "five_meal_calorie_mae_by_model_variant.csv",
    index=False,
)

print(mae_by_variant.to_string(index=False))
print()
print(condition_summary.to_string(index=False))
