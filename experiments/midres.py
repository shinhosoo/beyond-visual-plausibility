"""Inject the refined features back into the mid-level backbone representation."""
import argparse
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from SkinCancer.configs.model_config import BACKBONE_NAME, D_ATTN, IMG_SIZE, N_HEADS, N_SLOTS, SLOT_AGG
from SkinCancer.configs.training_config import (
    ACCUMULATION_STEPS, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY,
)
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.models.teme import LightweightTopoBlock, MultiScaleMicroEncoder
from SkinCancer.utils.seed import set_seed
from experiments.ltv_fix import LTVFixed
from experiments.residual import get_data, inv_freq, macro_auc

OUT = os.path.join(OUTPUT_ROOT, "residual")
os.makedirs(OUT, exist_ok=True)
GRID = 40

class MidRes(nn.Module):
    def __init__(self, mode="A", pretrained=True):
        super().__init__()
        self.mode = mode
        self.net = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=7)
        if hasattr(self.net, "set_grad_checkpointing"):
            self.net.set_grad_checkpointing(True)
        with torch.no_grad():
            h = self.net.stem(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
            h = self.net.stages[0](h); h = self.net.stages[1](h)
        assert h.shape[1] == h.shape[2], f"NHWC : {tuple(h.shape)}"
        c = h.shape[-1]
        self.c = c
        self.micro_proj = nn.Sequential(
            nn.Conv2d(c, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            MultiScaleMicroEncoder(256, 256), LightweightTopoBlock(256, reduced_dim=32),
            nn.AdaptiveAvgPool2d(GRID))
        self.ltv = LTVFixed(d_macro=c, d_micro=256, d_attn=D_ATTN, n_heads=N_HEADS,
                            n_slots=N_SLOTS, dropout=0.1, slot_agg=SLOT_AGG, mode="bias")
        if mode == "A":
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.proj = nn.Linear(D_ATTN, c)
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        h = self.net.stem(x)
        h = self.net.stages[0](h)
        h = self.net.stages[1](h)
        hw = h.permute(0, 3, 1, 2).contiguous()
        macro = hw.mean(dim=(2, 3))
        micro = self.micro_proj(hw)
        fused, sal = self.ltv(macro, micro)
        att = sal.mean(1).max(1).values
        att = att / att.max(dim=1, keepdim=True).values.clamp_min(1e-12)
        att = Fn.interpolate(att.view(-1, 1, GRID, GRID).float(), size=h.shape[1:3],
                             mode="bilinear", align_corners=False)[:, 0]
        if self.mode == "A":
            h = h * (1 + self.gamma * att.unsqueeze(-1).to(h.dtype))
        else:
            h = h + self.proj(fused).unsqueeze(1).unsqueeze(1) * att.unsqueeze(-1).to(h.dtype)
        h = self.net.stages[2](h)
        h = self.net.stages[3](h)
        return self.net.forward_head(h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "B"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS_STAGE1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    tag = f"mid{a.mode}"
    tr, va, te = get_data(a.fold, build_image_path_map())
    if a.limit:
        tr = tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True)
    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    model = MidRes(a.mode, pretrained=not a.no_pretrained).to(DEVICE)
    bb = [p for n, p in model.named_parameters() if n.startswith("net.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("net.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": LEARNING_RATE},
                             {"params": hd, "lr": LEARNING_RATE * 10}],
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = make_loader(tr, "label", build_train_transform(), shuffle=True)
    vl = make_loader(va, "label", build_val_transform(), shuffle=False)
    print(f"[midres:{tag}_f{a.fold}_s{a.seed}] mode={a.mode} ={model.c} "
          f"train={len(tr)} val={len(va)} backbone {sum(p.numel() for p in bb):,} / {sum(p.numel() for p in hd):,}")

    ck, best = os.path.join(OUT, f"{tag}_f{a.fold}_s{a.seed}.pth"), -1.0
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (img, lbl) in enumerate(tl):
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = ce(model(img.to(DEVICE)), lbl.to(DEVICE).long())
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss)
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for img, lbl in vl:
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    P.append(torch.softmax(model(img.to(DEVICE)).float(), 1).cpu().numpy())
                Y.append(lbl.numpy().flatten())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        g = float(model.gamma) if a.mode == "A" else float(model.proj.weight.abs().mean())
        print(f"Epoch {ep+1:02d}/{a.epochs} | Loss {tot/max(len(tl),1):.4f} | Val AUC {macro_auc(P,Y):.4f} "
              f"Acc {(P.argmax(1)==Y).mean():.4f} | {g:.4f}")
        if macro_auc(P, Y) > best:
            best = macro_auc(P, Y); torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for img, lbl in vl:
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                P.append(torch.softmax(model(img.to(DEVICE)).float(), 1).cpu().numpy())
            Y.append(lbl.numpy().flatten())
    P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
    f = os.path.join(OUT, f"{tag}_f{a.fold}_s{a.seed}_val_predictions.npz")
    np.savez(f, y_prob=P, y_true=Y, image_id=va["image_id"].values)
    print(f"[midres:{tag}] fold {a.fold} val acc={(P.argmax(1)==Y).mean():.4f} : {f}")

if __name__ == "__main__":
    main()
