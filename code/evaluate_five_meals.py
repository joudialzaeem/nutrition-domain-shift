"""Run frozen source and MMD models on the 25 personal meal photographs."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from train_model import build_model, get_transforms              

MODELS = ("resnet50", "efficientnet_b0", "vit_b_16")
CONDITIONS = {
    1: "overhead_good_light",
    2: "dim_light",
    3: "side_angle",
    4: "partial_crop",
    5: "mixed_ingredients",
}
MEALS = {
    1: "pizza",
    2: "turkey_and_potatoes",
    3: "tuna_salad",
    4: "chicken_fricassee",
    5: "prawns_and_bulgur",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    images_dir = root / "five_meal_study" / "images" / "5 meals images"
    output_dir = root / "results" / "five_meal"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = {}
    for path in images_dir.iterdir():
        match = re.fullmatch(r"([1-5])\.([1-5])\.(?:jpg|jpeg|png)", path.name, re.I)
        if match:
            key = tuple(int(x) for x in match.groups())
            if key in image_files:
                raise ValueError(f"Duplicate meal/condition: {key}")
            image_files[key] = path
    expected = {(meal, condition) for meal in range(1, 6) for condition in range(1, 6)}
    if image_files.keys() != expected:
        raise ValueError(f"Missing: {sorted(expected - image_files.keys())}; unexpected: {sorted(image_files.keys() - expected)}")
    keys = sorted(image_files)
    print(f"Found {len(keys)} meal photographs", flush=True)

    with open(root / "data" / "target_scaler.json", encoding="utf-8") as f:
        scaler = json.load(f)
    targets = scaler["targets"]
    if targets != ["calories", "protein", "carbs", "fat"]:
        raise ValueError(f"Unexpected target order: {targets}")
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    results = []
    for model_name in MODELS:
        _, eval_transform = get_transforms(model_name)
        for variant, postfix in (("source", "best"), ("mmd", "mmd_best")):
            checkpoint = root / "checkpoints" / f"{model_name}_{postfix}.pt"
            print(f"Evaluating {checkpoint.name}", flush=True)
            model, _ = build_model(model_name)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            all_preds = []
            with torch.inference_mode():
                for start in range(0, len(keys), args.batch_size):
                    batch_keys = keys[start:start + args.batch_size]
                    tensors = []
                    for key in batch_keys:
                                                                                                
                        with Image.open(image_files[key]) as im:
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            tensors.append(eval_transform(im))
                    x = torch.stack(tensors).to(device)
                    y_std = model(x).cpu().numpy()
                    all_preds.extend(y_std * std + mean)

            for key, pred in zip(keys, all_preds):
                meal, condition = key
                row = {
                    "filename": image_files[key].name,
                    "meal_id": meal,
                    "meal": MEALS[meal],
                    "condition_id": condition,
                    "condition": CONDITIONS[condition],
                    "model": model_name,
                    "variant": variant,
                }
                row.update({f"pred_{target}": float(value) for target, value in zip(targets, pred)})
                results.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    output = output_dir / "five_meal_predictions.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    expected_rows = 5 * 5 * 3 * 2
    assert len(results) == expected_rows
    print(f"Saved {len(results)} predictions to {output}", flush=True)
    print("These are raw predictions; no ground-truth labels or post-hoc clipping used.", flush=True)


if __name__ == "__main__":
    main()
