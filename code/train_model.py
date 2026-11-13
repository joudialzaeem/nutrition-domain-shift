import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T


TARGETS = ["calories", "protein", "carbs", "fat"]
SEED = 42


class NutritionDataset(Dataset):
    def __init__(self, df, image_root, transform, mean, std):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(
            self.image_root / row["dish_id"] / "rgb.png"
        ).convert("RGB")

        image = self.transform(image)

        y_raw = torch.tensor(
            [float(row[c]) for c in TARGETS],
            dtype=torch.float32
        )

        y = (y_raw - self.mean) / self.std

        return {
            "image": image,
            "target": y,
            "target_raw": y_raw,
            "dish_id": row["dish_id"]
        }


def get_weights(name):
    if name == "resnet50":
        return models.ResNet50_Weights.DEFAULT
    if name == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.DEFAULT
    if name == "vit_b_16":
        return models.ViT_B_16_Weights.DEFAULT
    raise ValueError(name)


def get_transforms(name):
    weights = get_weights(name)
    eval_tf = weights.transforms()

    train_tf = T.Compose([
        T.RandomResizedCrop(
            224,
            scale=(0.9, 1.0),
            ratio=(0.9, 1.1),
            interpolation=eval_tf.interpolation,
            antialias=True
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(
            mean=eval_tf.mean,
            std=eval_tf.std
        )
    ])

    return train_tf, eval_tf


def build_model(name):
    weights = get_weights(name)

    if name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 4)
        head = model.fc

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 4
        )
        head = model.classifier

    elif name == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, 4
        )
        head = model.heads

    return model, head


def train_epoch(model, loader, optimizer, loss_fn, device, head_only):
    if head_only:
        model.eval()
        if hasattr(model, "fc"):
            model.fc.train()
        elif hasattr(model, "classifier"):
            model.classifier.train()
        else:
            model.heads.train()
    else:
        model.train()

    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, loss_fn, device):
    model.eval()
    total = 0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        total += loss.item() * x.size(0)

    return total / len(loader.dataset)


def fit_stage(
    model, loader_train, loader_val,
    optimizer, loss_fn, device,
    epochs, patience, head_only
):
    best_loss = float("inf")
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, loader_train, optimizer,
            loss_fn, device, head_only
        )

        val_loss = eval_loss(
            model, loader_val, loss_fn, device
        )

        history.append(
            [epoch, train_loss, val_loss]
        )

        print(
            f"epoch {epoch}: "
            f"train={train_loss:.4f}, "
            f"val={val_loss:.4f}",
            flush=True
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print("early stopping", flush=True)
            break

    model.load_state_dict(best_state)

    return model, history, best_loss


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
        ids.extend(batch["dish_id"])

    return torch.cat(preds), torch.cat(targets), ids


def metrics_table(y_true, y_pred):
    rows = []

    for i, name in enumerate(TARGETS):
        yt = y_true[:, i].numpy()
        yp = y_pred[:, i].numpy()

        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))

        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        rows.append({
            "target": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "resnet50",
            "efficientnet_b0",
            "vit_b_16"
        ],
        required=True
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    checkpoints_dir = project_dir / "checkpoints"

    results_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    df = pd.read_csv(
        data_dir / "nutrition5k_model_splits.csv"
    )

    with open(
        data_dir / "target_scaler.json"
    ) as f:
        scaler = json.load(f)

    mean = np.array(
        scaler["mean"],
        dtype=np.float32
    )

    std = np.array(
        scaler["std"],
        dtype=np.float32
    )

    train_df = df[df["model_split"] == "train"]
    val_df = df[df["model_split"] == "val"]
    test_df = df[df["model_split"] == "test"]

    assert len(train_df) == 2245
    assert len(val_df) == 484
    assert len(test_df) == 498

    train_tf, eval_tf = get_transforms(args.model)
    image_root = data_dir / "images"

    train_ds = NutritionDataset(
        train_df, image_root,
        train_tf, mean, std
    )

    val_ds = NutritionDataset(
        val_df, image_root,
        eval_tf, mean, std
    )

    test_ds = NutritionDataset(
        test_df, image_root,
        eval_tf, mean, std
    )

    batch_size = 16 if args.model == "vit_b_16" else 32

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print(
        "split sizes:",
        len(train_ds), len(val_ds), len(test_ds),
        flush=True
    )

    model, head = build_model(args.model)
    model = model.to(device)

    loss_fn = nn.MSELoss()

    for p in model.parameters():
        p.requires_grad = False

    for p in head.parameters():
        p.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )

    model, history1, stage1_val = fit_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        epochs=5,
        patience=3,
        head_only=True
    )

    stage1_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    for p in model.parameters():
        p.requires_grad = True

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4
    )

    model, history2, stage2_val = fit_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        epochs=15,
        patience=4,
        head_only=False
    )

    if stage1_val < stage2_val:
        model.load_state_dict(stage1_state)
        print("using stage 1 model", flush=True)
    else:
        print("using fine-tuned model", flush=True)

    torch.save(
        model.state_dict(),
        checkpoints_dir / f"{args.model}_best.pt"
    )

    pred, true, ids = predict(
        model,
        test_loader,
        device,
        mean,
        std
    )

    metrics = metrics_table(true, pred)

    metrics.to_csv(
        results_dir / f"{args.model}_test_metrics.csv",
        index=False
    )

    pred_df = pd.DataFrame({"dish_id": ids})

    for i, name in enumerate(TARGETS):
        pred_df[f"true_{name}"] = true[:, i].numpy()
        pred_df[f"pred_{name}"] = pred[:, i].numpy()

    pred_df.to_csv(
        results_dir / f"{args.model}_test_predictions.csv",
        index=False
    )

    history1 = pd.DataFrame(
        history1,
        columns=["epoch", "train_loss", "val_loss"]
    )
    history1["stage"] = "head"

    history2 = pd.DataFrame(
        history2,
        columns=["epoch", "train_loss", "val_loss"]
    )
    history2["stage"] = "finetune"

    history = pd.concat(
        [history1, history2],
        ignore_index=True
    )

    history.to_csv(
        results_dir / f"{args.model}_history.csv",
        index=False
    )

    print("\nTest metrics:")
    print(metrics)
    print("\ndone")


if __name__ == "__main__":
    main()
