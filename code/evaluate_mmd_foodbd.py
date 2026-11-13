import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from train_model import TARGETS, get_transforms, build_model, metrics_table


class FoodBDDataset(Dataset):
    def __init__(self, df, data_root, transform):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            self.data_root / row["image_path"]
        ).convert("RGB")

        image = self.transform(image)

        target = torch.tensor(
            [float(row[c]) for c in TARGETS],
            dtype=torch.float32
        )

        return {
            "image": image,
            "target_raw": target,
            "FoodImage": row["FoodImage"]
        }


@torch.no_grad()
def predict(model, loader, device, mean, std):
    model.eval()

    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)

    preds = []
    targets = []
    ids = []

    for batch in loader:
        x = batch["image"].to(device)

        pred_std = model(x).cpu()
        pred = pred_std * std + mean

        preds.append(pred)
        targets.append(batch["target_raw"])
        ids.extend(batch["FoodImage"])

    return torch.cat(preds), torch.cat(targets), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["resnet50", "efficientnet_b0", "vit_b_16"],
        required=True
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    checkpoints_dir = project_dir / "checkpoints"

    foodbd_dir = data_dir / "foodbd_eval_1833"

    df = pd.read_csv(
        foodbd_dir / "foodbd_evaluation_1833.csv"
    )

    df = df[df["foodbd_split"] == "test"].copy()

    assert len(df) == 180

    with open(data_dir / "target_scaler.json") as f:
        scaler = json.load(f)

    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)

    _, eval_tf = get_transforms(args.model)

    dataset = FoodBDDataset(
        df,
        foodbd_dir,
        eval_tf
    )

    batch_size = 16 if args.model == "vit_b_16" else 32

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda")

    model, _ = build_model(args.model)

    state_dict = torch.load(
        checkpoints_dir / f"{args.model}_mmd_best.pt",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    pred, true, ids = predict(
        model,
        loader,
        device,
        mean,
        std
    )

    metrics = metrics_table(true, pred)

    metrics.to_csv(
        results_dir / f"{args.model}_mmd_foodbd_test180_metrics.csv",
        index=False
    )

    pred_df = pd.DataFrame({"FoodImage": ids})

    for i, name in enumerate(TARGETS):
        pred_df[f"true_{name}"] = true[:, i].numpy()
        pred_df[f"pred_{name}"] = pred[:, i].numpy()

    pred_df.to_csv(
        results_dir / f"{args.model}_mmd_foodbd_test180_predictions.csv",
        index=False
    )

    print(metrics)


if __name__ == "__main__":
    main()
