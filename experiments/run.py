"""Train and evaluate the Stage 2 branches.

The predict stage writes the Stage 1 probability and both branch distributions, from which the
seven-class distribution is formed by soft routing.

  python -m experiments.run --stage mel     --tag baseline --teme 0 --ltv 0
  python -m experiments.run --stage nonmel  --tag baseline --teme 0 --ltv 0
  python -m experiments.run --stage predict --tag baseline --teme 0 --ltv 0 --split test
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from SkinCancer.configs.data_config import (
    FINAL_CLASS_NAMES, FINAL7_TEST_CSV, NM_CLASS_NAMES, SPLIT_ROOT,
    STAGE2_FINAL_MAPPING,
)
from SkinCancer.configs.training_config import (
    BATCH_SIZE, DEVICE, EPOCHS_STAGE2_MELANOCYTIC, EPOCHS_STAGE2_NONMELANOCYTIC,
    LEARNING_RATE, OUTPUT_ROOT, SEED, STAGE1_MODEL_PATH, WEIGHT_DECAY,
)
from SkinCancer.data.dataset import LocalSkinCancerDataset
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import (
    build_image_path_map, load_stage2_melanocytic_data,
    load_stage2_nonmelanocytic_data,
)
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.eval.evaluate import validate
from SkinCancer.models.stage1 import create_stage1_classifier
from SkinCancer.train.train_utils import RobustFocalLoss, train_one_epoch
from SkinCancer.utils.seed import set_seed
from ablation.models import build, describe

EXPERIMENTS_DIR = os.path.join(OUTPUT_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def ckpt(tag, branch):
    return os.path.join(EXPERIMENTS_DIR, f"{tag}_stage2_{branch}.pth")


def pred_path(tag, split):
    return os.path.join(EXPERIMENTS_DIR, f"{tag}_{split}_predictions.npz")


def train_branch(a):
    path_map = build_image_path_map()

    if a.stage == "mel":
        tr, va = load_stage2_melanocytic_data(path_map)
        n_cls, multiclass = 1, False
        epochs = a.epochs_mel or EPOCHS_STAGE2_MELANOCYTIC
        n_pos = int((tr["stage2_label"] == 1).sum())
        n_neg = int((tr["stage2_label"] == 0).sum())
        base_pw = n_neg / max(n_pos, 1)
        pw = {"orig": base_pw, "sqrt": base_pw ** 0.5, "one": 1.0}[a.pos_weight]
        if a.mel_loss == "focal":
            criterion = RobustFocalLoss(gamma=a.focal_gamma,
                                        alpha=[1.0 - a.focal_alpha, a.focal_alpha])
            desc = f"focal(gamma={a.focal_gamma}, alpha_pos={a.focal_alpha})"
        else:
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pw], dtype=torch.float32, device=DEVICE))
            desc = f"BCE(pos_weight={pw:.4f}, mode={a.pos_weight}, orig={base_pw:.4f})"
        sampler = None
        print(f"[exp:{a.tag}][mel] train={len(tr)} val={len(va)} loss={desc}")
    else:
        tr, va = load_stage2_nonmelanocytic_data(path_map)
        n_cls, multiclass = 5, True
        epochs = a.epochs_nonmel or EPOCHS_STAGE2_NONMELANOCYTIC
        counts = tr["stage2_label"].value_counts().sort_index().values
        w = torch.tensor(counts.sum() / (len(counts) * counts),
                         dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=w)
        sampler = None
        if a.nonmel_sampler == "weighted":
            inv = 1.0 / counts
            sw = inv[tr["stage2_label"].astype(int).values]
            sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(sw), replacement=True)
        print(f"[exp:{a.tag}][nonmel] train={len(tr)} val={len(va)} "
              f"sampler={a.nonmel_sampler} weights="
              + ", ".join(f"{c}:{v:.3f}" for c, v in zip(NM_CLASS_NAMES, w.tolist())))

    model = build(n_cls, a.teme, a.ltv, a.simple_head).to(DEVICE)
    print(f"[exp:{a.tag}] {describe(model)}  epochs={epochs} seed={a.seed}")

    if sampler is None:
        tl = make_loader(tr, "stage2_label", build_train_transform(), shuffle=True)
    else:
        ds = LocalSkinCancerDataset(tr["image_path"].tolist(),
                                    tr["stage2_label"].astype(float).tolist(),
                                    build_train_transform())
        tl = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=4, pin_memory=True)
    vl = make_loader(va, "stage2_label", build_val_transform(), shuffle=False)

    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    out, best = ckpt(a.tag, a.stage), -1.0
    for ep in range(epochs):
        loss = train_one_epoch(model, tl, criterion, opt, scaler, is_multiclass=multiclass,
                               epoch=ep + 1, total_epochs=epochs,
                               stage_name=f"exp:{a.tag}:{a.stage}")
        auc = validate(model, vl, criterion, is_multiclass=multiclass,
                       stage_name=f"exp:{a.tag}:{a.stage}", epoch=ep + 1,
                       total_epochs=epochs,
                       class_names=NM_CLASS_NAMES if multiclass else None)
        print(f"Epoch {ep+1:02d}/{epochs} | Loss: {loss:.4f} | Val AUC: {auc:.4f}")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out)
            print(f"  -> checkpoint saved (val AUC {best:.4f})")
    print(f"[exp:{a.tag}][{a.stage}] best val AUC = {best:.4f} -> {out}")


def load_split7(split, path_map):
    csv = FINAL7_TEST_CSV if split == "test" else os.path.join(SPLIT_ROOT, f"global_{split}.csv")
    df = pd.read_csv(csv)
    if "diagnosis" not in df.columns and "dx" in df.columns:
        df = df.rename(columns={"dx": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].str.lower()
    df["final_label"] = df["diagnosis"].map(STAGE2_FINAL_MAPPING)
    df["image_path"] = df["image_id"].map(path_map)
    return df.dropna(subset=["final_label", "image_path"]).reset_index(drop=True)


@torch.no_grad()
def predict(a):
    """Combine the Stage 1 probability and the branch distributions by soft routing.
"""
    path_map = build_image_path_map()
    df = load_split7(a.split, path_map)
    loader = make_loader(df, "final_label", build_val_transform(), shuffle=False)
    print(f"[exp:{a.tag}] split={a.split} n={len(df)}")

    s1 = create_stage1_classifier().to(DEVICE)
    s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE,
                                  weights_only=True), strict=True)
    mel = build(1, a.teme, a.ltv, a.simple_head).to(DEVICE)
    mel.load_state_dict(torch.load(ckpt(a.tag, "mel"), map_location=DEVICE,
                                   weights_only=True), strict=True)
    nm = build(5, a.teme, a.ltv, a.simple_head).to(DEVICE)
    nm.load_state_dict(torch.load(ckpt(a.tag, "nonmel"), map_location=DEVICE,
                                  weights_only=True), strict=True)
    for m in (s1, mel, nm):
        m.eval()

    P1, PM, PN, Y = [], [], [], []
    for img, lbl in tqdm(loader, desc=f"exp:{a.tag}:{a.split}", leave=False):
        img = img.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            p1 = torch.sigmoid(s1(img)).view(-1).float()
            pm = torch.sigmoid(mel(img)).view(-1).float()
            pn = torch.softmax(nm(img), dim=1).float()
        P1.extend(p1.cpu().numpy()); PM.extend(pm.cpu().numpy())
        PN.extend(pn.cpu().numpy()); Y.extend(lbl.numpy().flatten())

    np.savez(pred_path(a.tag, a.split),
             p_stage1=np.array(P1), p_mel=np.array(PM), p_nonmel=np.array(PN),
             y_true=np.array(Y).astype(int), image_id=df["image_id"].values)
    print(f"[exp:{a.tag}] : {pred_path(a.tag, a.split)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["mel", "nonmel", "predict"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--teme", type=int, default=1)
    ap.add_argument("--ltv", type=int, default=1)
    ap.add_argument("--simple-head", type=int, default=0)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--mel-loss", default="bce", choices=["bce", "focal"])
    ap.add_argument("--pos-weight", default="orig", choices=["orig", "sqrt", "one"])
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.5,
                    help="positive-class (MEL) alpha; Stage 1 uses 0.2")
    ap.add_argument("--nonmel-sampler", default="none", choices=["none", "weighted"])
    ap.add_argument("--epochs-mel", type=int, default=0)
    ap.add_argument("--epochs-nonmel", type=int, default=0)
    a = ap.parse_args()

    set_seed(a.seed)
    if a.stage == "predict":
        predict(a)
    else:
        train_branch(a)


if __name__ == "__main__":
    main()
