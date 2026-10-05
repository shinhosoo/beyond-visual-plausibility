"""Hierarchical design variants.

  --mode subset  branches trained on the corresponding subset of the training split
  --mode all     branches trained on all training images
"""

import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from SkinCancer.configs.data_config import FINAL7_TEST_CSV, SPLIT_ROOT
from SkinCancer.configs.model_config import BACKBONE_NAME
from SkinCancer.configs.training_config import (
    DEVICE, EPOCHS_STAGE2_MELANOCYTIC, EPOCHS_STAGE2_NONMELANOCYTIC,
    LEARNING_RATE, OUTPUT_ROOT, SEED, STAGE1_MODEL_PATH, WEIGHT_DECAY,
)
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.eval.evaluate import validate
from SkinCancer.models.stage1 import create_stage1_classifier
from SkinCancer.train.train_utils import train_one_epoch
from SkinCancer.utils.seed import set_seed
from ablation.models import build, describe

OUT = os.path.join(OUTPUT_ROOT, "hier")
os.makedirs(OUT, exist_ok=True)

MEL_IN = {"nv": 0, "mel": 1}
NM_IN = {"bkl": 0, "df": 1, "vasc": 2, "bcc": 3, "akiec": 4}
FINAL7 = {"nv": 0, "mel": 1, "bkl": 2, "df": 3, "vasc": 4, "bcc": 5, "akiec": 6}


def ckpt(tag, branch):
    return os.path.join(OUT, f"{tag}_stage2_{branch}.pth")


def load_global(split, path_map):
    csv = FINAL7_TEST_CSV if split == "test" else os.path.join(SPLIT_ROOT, f"global_{split}.csv")
    df = pd.read_csv(csv)
    if "diagnosis" not in df.columns and "dx" in df.columns:
        df = df.rename(columns={"dx": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].str.lower()
    df["image_path"] = df["image_id"].map(path_map)
    df["final_label"] = df["diagnosis"].map(FINAL7)
    return df.dropna(subset=["image_path", "final_label"]).reset_index(drop=True)


def branch_labels(df, branch, mode):
    inside = MEL_IN if branch == "mel" else NM_IN
    n_in = len(inside)
    if mode == "subset":
        d = df[df["diagnosis"].isin(inside)].copy()
        d["hlabel"] = d["diagnosis"].map(inside)
        return d.reset_index(drop=True), n_in
    d = df.copy()
    d["hlabel"] = d["diagnosis"].map(inside).fillna(n_in).astype(int)
    return d, n_in + 1


def transfer_stage1(model):
    s1 = create_stage1_classifier(backbone_name=BACKBONE_NAME, pretrained=False)
    s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location="cpu", weights_only=True), strict=True)
    r = model.backbone.load_state_dict(s1.state_dict(), strict=False)
    n = len(model.backbone.state_dict()) - len(r.missing_keys)
    print(f"[hier] Stage 1 -> branch backbone : {n} , "
          f"missing={r.missing_keys}, excluded={r.unexpected_keys}")


def train_branch(a):
    pm = build_image_path_map()
    tr, n_cls = branch_labels(load_global("train", pm), a.stage, a.mode)
    va, _ = branch_labels(load_global("val", pm), a.stage, a.mode)
    if a.limit:
        k = max(2, a.limit // n_cls)
        tr = tr.groupby("hlabel").head(k).reset_index(drop=True)
        va = va.groupby("hlabel").head(k).reset_index(drop=True)

    counts = tr["hlabel"].value_counts().reindex(range(n_cls), fill_value=0).values
    w = torch.tensor(counts.sum() / (n_cls * np.maximum(counts, 1)), dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    names = (list(MEL_IN) if a.stage == "mel" else list(NM_IN)) + (["other"] if a.mode == "allother" else [])
    print(f"[hier:{a.tag}][{a.stage}] mode={a.mode} init={a.init} train={len(tr)} val={len(va)} "
          f"classes={n_cls} counts={dict(zip(names, counts.tolist()))}")
    print(f"[hier:{a.tag}][{a.stage}] weights=" + ", ".join(f"{n}:{v:.3f}" for n, v in zip(names, w.tolist())))

    model = build(n_cls, a.teme, a.ltv, False).to(DEVICE)
    if a.init == "stage1":
        transfer_stage1(model)
    print(f"[hier:{a.tag}] {describe(model)}")

    epochs = a.epochs or (EPOCHS_STAGE2_MELANOCYTIC if a.stage == "mel" else EPOCHS_STAGE2_NONMELANOCYTIC)
    tl = make_loader(tr, "hlabel", build_train_transform(), shuffle=True)
    vl = make_loader(va, "hlabel", build_val_transform(), shuffle=False)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    out, best = ckpt(a.tag, a.stage), -1.0
    for ep in range(epochs):
        loss = train_one_epoch(model, tl, criterion, opt, scaler, is_multiclass=True,
                               epoch=ep + 1, total_epochs=epochs, stage_name=f"hier:{a.tag}:{a.stage}")
        auc = validate(model, vl, criterion, is_multiclass=True, stage_name=f"hier:{a.tag}:{a.stage}",
                       epoch=ep + 1, total_epochs=epochs, class_names=names)
        print(f"Epoch {ep+1:02d}/{epochs} | Loss: {loss:.4f} | Val AUC: {auc:.4f}")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out)
            print(f"  -> checkpoint saved (val AUC {best:.4f})")
    print(f"[hier:{a.tag}][{a.stage}] best val AUC = {best:.4f} -> {out}")


@torch.no_grad()
def run_loader(models, loader):
    outs = [[] for _ in models]
    ys = []
    for img, lbl in tqdm(loader, leave=False):
        img = img.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            res = [f(m(img)) for m, f in models]
        for i, r in enumerate(res):
            outs[i].append(r.float().cpu().numpy())
        ys.append(lbl.numpy().flatten())
    return [np.concatenate(o) for o in outs], np.concatenate(ys).astype(int)


def predict(a):
    pm = build_image_path_map()
    df = load_global(a.split, pm)
    loader = make_loader(df, "final_label", build_val_transform(), shuffle=False)
    n_mel = 3 if a.mode == "allother" else 2
    n_nm = 6 if a.mode == "allother" else 5

    s1 = create_stage1_classifier(backbone_name=BACKBONE_NAME, pretrained=False).to(DEVICE)
    s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=True)
    mel = build(n_mel, a.teme, a.ltv, False).to(DEVICE)
    mel.load_state_dict(torch.load(ckpt(a.tag, "mel"), map_location=DEVICE, weights_only=True), strict=True)
    nm = build(n_nm, a.teme, a.ltv, False).to(DEVICE)
    nm.load_state_dict(torch.load(ckpt(a.tag, "nonmel"), map_location=DEVICE, weights_only=True), strict=True)
    for m in (s1, mel, nm):
        m.eval()

    (p1, pa, pc), y = run_loader(
        [(s1, lambda z: torch.sigmoid(z).view(-1)),
         (mel, lambda z: torch.softmax(z, 1)),
         (nm, lambda z: torch.softmax(z, 1))], loader)
    f = os.path.join(OUT, f"{a.tag}_{a.split}_predictions.npz")
    np.savez(f, p_stage1=p1, p_mel_branch=pa, p_nonmel_branch=pc, y_true=y,
             image_id=df["image_id"].values, mode=a.mode)
    print(f"[hier:{a.tag}] {a.split} n={len(y)} : {f}")


def flatpred(a):
    pm = build_image_path_map()
    df = load_global(a.split, pm)
    loader = make_loader(df, "final_label", build_val_transform(), shuffle=False)
    m = timm.create_model(BACKBONE_NAME, pretrained=False, num_classes=7).to(DEVICE)
    path = a.flat_ckpt or os.path.join(OUTPUT_ROOT, "onestage_7class.pth")
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=True)
    m.eval()
    (P,), y = run_loader([(m, lambda z: torch.softmax(z, 1))], loader)
    f = os.path.join(OUT, f"flat_{a.split}_predictions.npz")
    np.savez(f, y_prob=P, y_true=y, image_id=df["image_id"].values)
    print(f"[hier:flat] {a.split} n={len(y)} acc={(P.argmax(1)==y).mean():.4f} : {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["mel", "nonmel", "predict", "flatpred"])
    ap.add_argument("--tag", default="hier")
    ap.add_argument("--mode", default="allother", choices=["subset", "allother"])
    ap.add_argument("--init", default="stage1", choices=["imagenet", "stage1"])
    ap.add_argument("--teme", type=int, default=1)
    ap.add_argument("--ltv", type=int, default=1)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flat-ckpt", default="")
    a = ap.parse_args()
    set_seed(a.seed)
    {"mel": train_branch, "nonmel": train_branch, "predict": predict, "flatpred": flatpred}[a.stage](a)


if __name__ == "__main__":
    main()
