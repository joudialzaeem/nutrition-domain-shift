import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from train_model import (
    TARGETS,
    NutritionDataset,
    get_transforms,
    build_model
)


class FoodBDUnlabeledDataset(Dataset):
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

        return {
            "image": image,
            "FoodImage": row["FoodImage"]
        }


def get_feature_layer(model, name):
    if name == "resnet50":
        return model.fc

    if name == "efficientnet_b0":
        return model.classifier[1]

    if name == "vit_b_16":
        return model.heads.head

    raise ValueError(name)


class FeatureCapture:
    def __init__(self, layer):
        self.features = None

        self.hook = layer.register_forward_pre_hook(
            self._hook
        )

    def _hook(self, module, inputs):
        self.features = inputs[0]

    def close(self):
        self.hook.remove()


def rbf_kernel(x, y, sigma):
    dist2 = torch.cdist(
        x,
        y,
        p=2
    ) ** 2

    return torch.exp(
        -dist2 / (2 * sigma ** 2)
    )


def mmd_loss(source_features, target_features):
                                                           
                                          
    source_features = nn.functional.normalize(
        source_features,
        dim=1
    )

    target_features = nn.functional.normalize(
        target_features,
        dim=1
    )

    sigmas = [0.5, 1.0, 2.0]

    loss = 0.0

    for sigma in sigmas:
        k_ss = rbf_kernel(
            source_features,
            source_features,
            sigma
        )

        k_tt = rbf_kernel(
            target_features,
            target_features,
            sigma
        )

        k_st = rbf_kernel(
            source_features,
            target_features,
            sigma
        )

        loss = loss + (
            k_ss.mean()
            + k_tt.mean()
            - 2 * k_st.mean()
        )

    return loss / len(sigmas)


@torch.no_grad()
def validation_loss(
    model,
    loader,
    loss_fn,
    device
):
    model.eval()

    total = 0.0

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["target"].to(device)

        pred = model(x)

        loss = loss_fn(
            pred,
            y
        )

        total += (
            loss.item()
            * x.size(0)
        )

    return (
        total /
        len(loader.dataset)
    )


def train_epoch(
    model,
    source_loader,
    target_loader,
    optimizer,
    regression_loss_fn,
    feature_capture,
    device,
    lambda_mmd
):
    model.train()

    total_reg = 0.0
    total_mmd = 0.0
    total_n = 0

    target_iter = iter(target_loader)

    for source_batch in source_loader:

        try:
            target_batch = next(target_iter)

        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        x_source = source_batch["image"].to(device)
        y_source = source_batch["target"].to(device)

        x_target = target_batch["image"].to(device)

        optimizer.zero_grad()

        pred_source = model(x_source)
        source_features = feature_capture.features

        reg_loss = regression_loss_fn(
            pred_source,
            y_source
        )

        _ = model(x_target)
        target_features = feature_capture.features

        domain_loss = mmd_loss(
            source_features,
            target_features
        )

        loss = (
            reg_loss
            + lambda_mmd * domain_loss
        )

        loss.backward()
        optimizer.step()

        batch_n = x_source.size(0)

        total_reg += (
            reg_loss.item()
            * batch_n
        )

        total_mmd += (
            domain_loss.item()
            * batch_n
        )

        total_n += batch_n

    return (
        total_reg / total_n,
        total_mmd / total_n
    )


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

    project_dir = Path(
        __file__
    ).resolve().parents[1]

    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    checkpoints_dir = project_dir / "checkpoints"

                             
    n5k = pd.read_csv(
        data_dir /
        "nutrition5k_model_splits.csv"
    )

    train_df = n5k[
        n5k["model_split"] == "train"
    ]

    val_df = n5k[
        n5k["model_split"] == "val"
    ]

    assert len(train_df) == 2245
    assert len(val_df) == 484

                               
    foodbd_dir = (
        data_dir /
        "foodbd_eval_1833"
    )

    foodbd = pd.read_csv(
        foodbd_dir /
        "foodbd_evaluation_1833.csv"
    )

    target_df = foodbd[
        foodbd["foodbd_split"].isin(
            ["train", "valid"]
        )
    ].copy()

    test_df = foodbd[
        foodbd["foodbd_split"] == "test"
    ]

    assert len(target_df) == 1653
    assert len(test_df) == 180

                               
    with open(
        data_dir /
        "target_scaler.json"
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

    train_tf, eval_tf = get_transforms(
        args.model
    )

    source_train_ds = NutritionDataset(
        train_df,
        data_dir / "images",
        train_tf,
        mean,
        std
    )

    source_val_ds = NutritionDataset(
        val_df,
        data_dir / "images",
        eval_tf,
        mean,
        std
    )

    target_ds = FoodBDUnlabeledDataset(
        target_df,
        foodbd_dir,
        train_tf
    )

    batch_size = (
        16
        if args.model == "vit_b_16"
        else 32
    )

    source_train_loader = DataLoader(
        source_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    source_val_loader = DataLoader(
        source_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    target_loader = DataLoader(
        target_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU not available"
        )

    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True
    )

    print(
        "source train:",
        len(source_train_ds),
        flush=True
    )

    print(
        "source val:",
        len(source_val_ds),
        flush=True
    )

    print(
        "unlabeled FoodBD adaptation:",
        len(target_ds),
        flush=True
    )

    model, _ = build_model(
        args.model
    )

    checkpoint = (
        checkpoints_dir /
        f"{args.model}_best.pt"
    )

    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)

    feature_layer = get_feature_layer(
        model,
        args.model
    )

    feature_capture = FeatureCapture(
        feature_layer
    )

    regression_loss_fn = nn.MSELoss()

                                               
                                              
    optimizer = AdamW(
        model.parameters(),
        lr=1e-5
    )

    lambda_mmd = 0.1
    max_epochs = 8
    patience = 3

    best_val = float("inf")
    best_state = None
    wait = 0

    history = []

    for epoch in range(
        1,
        max_epochs + 1
    ):

        reg_loss, domain_loss = train_epoch(
            model,
            source_train_loader,
            target_loader,
            optimizer,
            regression_loss_fn,
            feature_capture,
            device,
            lambda_mmd
        )

        val_loss = validation_loss(
            model,
            source_val_loader,
            regression_loss_fn,
            device
        )

        history.append({
            "epoch": epoch,
            "source_reg_loss": reg_loss,
            "mmd_loss": domain_loss,
            "source_val_loss": val_loss
        })

        print(
            f"epoch {epoch}: "
            f"reg={reg_loss:.4f}, "
            f"mmd={domain_loss:.4f}, "
            f"val={val_loss:.4f}",
            flush=True
        )

        if val_loss < best_val:

            best_val = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            wait = 0

        else:
            wait += 1

        if wait >= patience:
            print(
                "early stopping",
                flush=True
            )
            break

    feature_capture.close()

    model.load_state_dict(
        best_state
    )

    output_checkpoint = (
        checkpoints_dir /
        f"{args.model}_mmd_best.pt"
    )

    torch.save(
        model.state_dict(),
        output_checkpoint
    )

    history = pd.DataFrame(
        history
    )

    history.to_csv(
        results_dir /
        f"{args.model}_mmd_history.csv",
        index=False
    )

    print(
        "\nbest source validation loss:",
        best_val
    )

    print(
        "saved:",
        output_checkpoint
    )

    print("done")


if __name__ == "__main__":
    main()
