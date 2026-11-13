import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "image_diagnostics"
FIGURES = ROOT / "figures"
IMAGES_ZIP = ROOT / "data" / "foodbd_test_images.zip"

FIGURES.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(
    RESULTS / "foodbd_test180_verified_image_features_and_errors.csv"
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    data.brightness,
    data.avg_calories_ae,
    s=19,
    alpha=0.6,
)
ax.set(
    xlabel="Mean grayscale brightness (0–1)",
    ylabel="Mean absolute calorie error across models (kcal)",
    title="FoodBD held-out test (n=180): brightness vs calorie error",
)
fig.tight_layout()
fig.savefig(
    FIGURES / "brightness_vs_calorie_error.png",
    dpi=180,
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    data.brightness,
    data.avg_protein_ae,
    s=19,
    alpha=0.6,
)
ax.set(
    xlabel="Mean grayscale brightness (0–1)",
    ylabel="Mean absolute protein error across models (g)",
    title="FoodBD held-out test (n=180): brightness vs protein error",
)
fig.tight_layout()
fig.savefig(
    FIGURES / "brightness_vs_protein_error.png",
    dpi=180,
)
plt.close(fig)

chosen = [
    "FoodBD-3299.jpg",
    "FoodBD-3519.jpg",
    "FoodBD-2995.jpg",
    "FoodBD-3220.jpg",
]

ordered = data.sort_values("avg_calories_ae")

for k in range(4):
    segment = ordered.iloc[k * 45:(k + 1) * 45]
    chosen.extend(
        segment.iloc[[10, 30]].FoodImage.tolist()
    )

chosen = list(dict.fromkeys(chosen))

with zipfile.ZipFile(IMAGES_ZIP) as archive:
    width = 256
    height = 256

    canvas = Image.new(
        "RGB",
        (3 * width, 4 * (height + 45)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for j, filename in enumerate(chosen[:12]):
        img = Image.open(
            io.BytesIO(
                archive.read("images/" + filename)
            )
        ).convert("RGB")

        img.thumbnail((width - 6, height - 6))

        x = (j % 3) * width + (width - img.width) // 2
        y = (j // 3) * (height + 45) + (height - img.height) // 2

        canvas.paste(img, (x, y))

        row = data.loc[
            data.FoodImage == filename
        ].iloc[0]

        draw.text(
            (
                (j % 3) * width + 6,
                (j // 3) * (height + 45) + height + 2,
            ),
            filename,
            fill="black",
        )

        draw.text(
            (
                (j % 3) * width + 6,
                (j // 3) * (height + 45) + height + 18,
            ),
            f"Calorie abs error: {row.avg_calories_ae:.0f} kcal",
            fill="black",
        )

    canvas.save(
        FIGURES / "case_inspection_contact_sheet.jpg",
        quality=92,
    )

print("selection", chosen[:12])
print("figures created")
