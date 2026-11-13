import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "image_diagnostics"
IMAGES_ZIP = ROOT / "data" / "foodbd_test_images.zip"

joined = pd.read_csv(RESULTS / "foodbd_test180_image_features_and_errors.csv")

with zipfile.ZipFile(IMAGES_ZIP) as archive:
    available = set(archive.namelist())
    assert len(joined) == joined.FoodImage.nunique() == 180
    assert set("images/" + n for n in joined.FoodImage).issubset(available)

    features = []

    for filename in joined.FoodImage:
        img = Image.open(
            io.BytesIO(archive.read("images/" + filename))
        ).convert("RGB")

        rgb = np.asarray(
            img.resize((224, 224), Image.Resampling.BILINEAR)
        )

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200) > 0

        mask = np.zeros_like(edges, dtype=bool)
        mask[:22, :] = True
        mask[-22:, :] = True
        mask[:, :22] = True
        mask[:, -22:] = True

        features.append({
            "FoodImage": filename,
            "brightness": gray.mean() / 255.0,
            "contrast": gray.std() / 255.0,
            "laplacian_variance": cv2.Laplacian(
                gray, cv2.CV_64F
            ).var(),
            "dark_pixel_fraction": (gray < 50).mean(),
            "edge_density": edges.mean(),
            "border_edge_density": edges[mask].mean(),
        })

features = pd.DataFrame(features)

old_features = [
    "brightness",
    "contrast",
    "sharpness",
    "saturation",
    "border_edge_density",
    "center_edge_density",
]

data = (
    joined.drop(columns=old_features)
    .merge(features, on="FoodImage", validate="one_to_one")
)

targets = ["calories", "protein", "carbs", "fat"]
models = ["ResNet50", "EfficientNet-B0", "ViT-B/16"]

for target in targets:
    actual = np.mean(
        [data[f"{model}_{target}_ae"].values for model in models],
        axis=0,
    )
    np.testing.assert_allclose(
        actual,
        data[f"avg_{target}_ae"].values,
        atol=1e-5,
    )

rows = []

for feature in [c for c in features.columns if c != "FoodImage"]:
    for target in targets:
        rho, p = spearmanr(
            data[feature],
            data[f"avg_{target}_ae"],
        )
        rows.append({
            "feature": feature,
            "target": target,
            "rho": rho,
            "p_raw": p,
        })

correlations = pd.DataFrame(rows)
correlations["p_holm"] = multipletests(
    correlations["p_raw"],
    method="holm",
)[1]
correlations["holm_sig_0.05"] = correlations["p_holm"] < 0.05

model_rows = []

for feature in [
    "brightness",
    "dark_pixel_fraction",
    "laplacian_variance",
]:
    for target in ["calories", "protein"]:
        for model in models:
            rho, p = spearmanr(
                data[feature],
                data[f"{model}_{target}_ae"],
            )
            model_rows.append({
                "feature": feature,
                "target": target,
                "model": model,
                "rho": rho,
                "p_raw": p,
            })

per_model = pd.DataFrame(model_rows)

quartile_rows = []

for feature in ["brightness", "dark_pixel_fraction"]:
    quartiles = pd.qcut(
        data[feature],
        4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    )

    for quartile, part in data.groupby(quartiles, observed=True):
        quartile_rows.append({
            "feature": feature,
            "quartile": quartile,
            "n": len(part),
            "avg_calorie_mae": part.avg_calories_ae.mean(),
            "avg_protein_mae": part.avg_protein_ae.mean(),
            "median_calorie_ae": part.avg_calories_ae.median(),
            "median_protein_ae": part.avg_protein_ae.median(),
        })

quartile_results = pd.DataFrame(quartile_rows)

sensitivity_rows = []

for excluded in [0, 5, 10]:
    subset = data.nsmallest(
        180 - excluded,
        "avg_calories_ae",
    )

    for feature in ["brightness", "dark_pixel_fraction"]:
        for target in ["calories", "protein"]:
            rho, p = spearmanr(
                subset[feature],
                subset[f"avg_{target}_ae"],
            )

            sensitivity_rows.append({
                "excluded_largest_calorie_errors": excluded,
                "feature": feature,
                "target": target,
                "rho": rho,
                "p_raw": p,
                "n": len(subset),
            })

sensitivity = pd.DataFrame(sensitivity_rows)

RESULTS.mkdir(parents=True, exist_ok=True)

data.to_csv(
    RESULTS / "foodbd_test180_verified_image_features_and_errors.csv",
    index=False,
)
correlations.to_csv(
    RESULTS / "foodbd_verified_spearman_holm_24_tests.csv",
    index=False,
)
per_model.to_csv(
    RESULTS / "foodbd_verified_per_model_correlations.csv",
    index=False,
)
quartile_results.to_csv(
    RESULTS / "foodbd_verified_image_feature_quartiles.csv",
    index=False,
)
sensitivity.to_csv(
    RESULTS / "foodbd_verified_outlier_sensitivity.csv",
    index=False,
)

print("MATCHED:", len(data), "UNIQUE:", data.FoodImage.nunique())
print(correlations.to_string(index=False))
