from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

models = ["resnet50", "efficientnet_b0", "vit_b_16"]
targets = ["calories", "protein", "carbs", "fat"]

rows = []

for model in models:
    before = pd.read_csv(RESULTS / f"{model}_foodbd_predictions.csv")
    before = before[before["foodbd_split"] == "test"].copy()
    after = pd.read_csv(RESULTS / f"{model}_mmd_foodbd_test180_predictions.csv")

    merged = before.merge(
        after,
        on="FoodImage",
        suffixes=("_before", "_after"),
        validate="one_to_one"
    )

    assert len(merged) == 180

    for target in targets:
        true_before = merged[f"true_{target}_before"].to_numpy(dtype=float)
        true_after = merged[f"true_{target}_after"].to_numpy(dtype=float)
        np.testing.assert_allclose(true_before, true_after)

        error_before = np.abs(
            true_before - merged[f"pred_{target}_before"].to_numpy(dtype=float)
        )
        error_after = np.abs(
            true_after - merged[f"pred_{target}_after"].to_numpy(dtype=float)
        )

        change = error_after - error_before

        rng = np.random.default_rng(42)
        bootstrap_means = np.empty(10000, dtype=float)

        for i in range(10000):
            idx = rng.integers(0, len(change), len(change))
            bootstrap_means[i] = change[idx].mean()

        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])

        _, p_raw = wilcoxon(
            error_after,
            error_before,
            alternative="two-sided",
            zero_method="wilcox"
        )

        before_mae = error_before.mean()
        after_mae = error_after.mean()
        mean_change = after_mae - before_mae

        rows.append({
            "model": model,
            "target": target,
            "before_mae": before_mae,
            "after_mae": after_mae,
            "mean_change": mean_change,
            "percent_change": 100.0 * mean_change / before_mae,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "wilcoxon_p_raw": p_raw
        })

result = pd.DataFrame(rows)
result["wilcoxon_p_holm"] = multipletests(
    result["wilcoxon_p_raw"],
    method="holm"
)[1]
result["holm_significant_0.05"] = result["wilcoxon_p_holm"] < 0.05

result.to_csv(
    RESULTS / "mmd_before_after_statistics_reproduced.csv",
    index=False
)

print(result.to_string(index=False))
