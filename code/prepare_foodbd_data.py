from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FOODBD = DATA / "foodbd"

metadata = pd.read_csv(FOODBD / "FoodBD_Meta_data.csv")
nutrition = pd.read_csv(FOODBD / "MealNutrition1837_All.csv")

split_lookup = (
    metadata[["filename", "types"]]
    .drop_duplicates(subset="filename", keep=False)
    .rename(columns={"filename": "FoodImage", "types": "foodbd_split"})
)

foodbd = nutrition.merge(split_lookup, on="FoodImage", how="inner")

foodbd = foodbd.rename(columns={
    "Calorie": "calories",
    "Protein": "protein",
    "CHO": "carbs",
    "Fat": "fat"
})

foodbd["image_path"] = "images/" + foodbd["FoodImage"]

foodbd = foodbd[
    [
        "FoodImage",
        "image_path",
        "foodbd_split",
        "calories",
        "protein",
        "carbs",
        "fat"
    ]
].reset_index(drop=True)

assert len(foodbd) == 1833
assert foodbd["FoodImage"].is_unique
assert foodbd["foodbd_split"].value_counts().to_dict() == {
    "train": 1288,
    "valid": 365,
    "test": 180
}

missing_labels = sorted(set(nutrition["FoodImage"]) - set(foodbd["FoodImage"]))
assert missing_labels == [
    "FoodBD-2282.jpg",
    "FoodBD-2383.jpg",
    "FoodBD-3381.jpg",
    "FoodBD-3508.jpg"
]

FOODBD.mkdir(parents=True, exist_ok=True)
OUTPUT = DATA / "foodbd_eval_1833"
OUTPUT.mkdir(parents=True, exist_ok=True)
foodbd.to_csv(OUTPUT / "foodbd_evaluation_1833.csv", index=False)

print("FoodBD evaluation meals:", len(foodbd))
print(foodbd["foodbd_split"].value_counts().to_string())
print("Nutrition-labelled images without matching dataset metadata:", missing_labels)
