"""Class-aware augmentation (cross-validated).

The strength depends only on class frequency, so no label information leaks across images.
"""
import argparse
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from SkinCancer.configs.model_config import BACKBONE_NAME, IMG_SIZE
from SkinCancer.configs.training_config import (
    ACCUMULATION_STEPS, BATCH_SIZE, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY,
)
from SkinCancer.utils.seed import set_seed
from SkinCancer.data.split_io import build_image_path_map
from experiments.residual import get_data, inv_freq, macro_auc

OUT = os.path.join(OUTPUT_ROOT, "residual")
os.makedirs(OUT, exist_ok=True)
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
GROUPS = {"flat": list(range(7)), "mel": [0, 1], "nonmel": [2, 3, 4, 5, 6]}


class AugSet(Dataset):
    def __init__(self, df, classes, strength, train):
        self.p = df["image_path"].tolist()
        self.y = [classes.index(int(v)) for v in df["label"]]
        self.ids = df["image_id"].tolist()
        self.train, self.f = train, strength

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        y = self.y[i]
        if self.train:
            f = self.f[y]
            if np.random.rand() < 0.5:
                img = TF.hflip(img)
            if np.random.rand() < 0.2:
                img = TF.vflip(img)
            img = TF.rotate(img, float(np.random.uniform(-30 * f, 30 * f)),
                            interpolation=TF.InterpolationMode.BILINEAR)
            if f > 1.01:
                s = 1 - 0.08 * f
                img = transforms.RandomResizedCrop(IMG_SIZE, scale=(max(s, 0.4), 1.0),
                                                   ratio=(0.85, 1.18))(img)
            j = min(0.1 * f, 0.4)
            img = transforms.ColorJitter(brightness=j, contrast=j)(img)
        return NORM(TF.to_tensor(img)), y


def strengths(counts, on):
    if not on:
        return np.ones(len(counts))
    return np.clip(np.sqrt(counts.max() / np.maximum(counts, 1)), 1.0, 4.0)


def mix_batch(x, y, minority, alpha=0.4):
    """ returns x and (y_a, y_b, lam_per_sample)"""
    n = x.shape[0]
    idx = torch.randperm(n, device=x.device)
    ok = minority[y] | minority[y[idx]]
    lam = torch.full((n,), 1.0, device=x.device)
    if ok.any():
        l = float(np.random.beta(alpha, alpha))
        l = max(l, 1 - l)
        lam[ok] = l
    xm = lam.view(-1, 1, 1, 1) * x + (1 - lam).view(-1, 1, 1, 1) * x[idx]
    return xm, y, y[idx], lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(GROUPS))
    ap.add_argument("--aug", required=True, choices=["none", "S", "M", "B"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS_STAGE1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    cls = GROUPS[a.model]
    tag = {"flat": "flat", "mel": "mel", "nonmel": "nm"}[a.model] + a.aug
    tr, va, _ = get_data(a.fold, build_image_path_map())
    tr = tr[tr["label"].astype(int).isin(cls)].reset_index(drop=True)
    va = va[va["label"].astype(int).isin(cls)].reset_index(drop=True)
    if a.limit:
        tr = tr.groupby("label").head(a.limit // len(cls) + 1).reset_index(drop=True)
    counts = np.array([(tr["label"].astype(int) == c).sum() for c in cls])
    st = strengths(counts, a.aug in ("S", "B"))
    use_mix = a.aug in ("M", "B")
    minority = torch.tensor([c != 0 for c in cls], device=DEVICE)
    print(f"[aug_cv:{tag}_f{a.fold}_s{a.seed}] model={a.model} aug={a.aug} "
          f"train={len(tr)} val={len(va)} classes={len(cls)}")
    print(" table / augmentation: " + ", ".join(f"{c}:{n}/{f:.2f}" for c, n, f in zip(cls, counts, st)))

    model = timm.create_model(BACKBONE_NAME, pretrained=not a.no_pretrained, num_classes=len(cls)).to(DEVICE)
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    ce = nn.CrossEntropyLoss(weight=inv_freq(counts), reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = DataLoader(AugSet(tr, cls, st, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(AugSet(va, cls, st, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    ck, best = os.path.join(OUT, f"{tag}_f{a.fold}_s{a.seed}.pth"), -1.0
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tl):
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            if use_mix:
                x, ya, yb, lam = mix_batch(x, y, minority)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                z = model(x)
                loss = (lam * ce(z, ya) + (1 - lam) * ce(z, yb)).mean() if use_mix else ce(z, y).mean()
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss)
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in vl:
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    P.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu().numpy())
                Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        auc = macro_auc(P, Y)
        print(f"Epoch {ep+1:02d}/{a.epochs} | Loss {tot/max(len(tl),1):.4f} | Val AUC {auc:.4f} "
              f"Acc {(P.argmax(1)==Y).mean():.4f}")
        if auc > best:
            best = auc; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in vl:
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                P.append(torch.softmax(model(x.to(DEVICE)).float(), 1).cpu().numpy())
            Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
    f = os.path.join(OUT, f"{tag}_f{a.fold}_s{a.seed}_val_predictions.npz")
    np.savez(f, y_prob=P, y_true=Y, image_id=va["image_id"].values, classes=np.array(cls))
    print(f"[aug_cv:{tag}] fold {a.fold} val acc={(P.argmax(1)==Y).mean():.4f} : {f}")


if __name__ == "__main__":
    main()
