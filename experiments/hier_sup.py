"""Hierarchical correction with mask-supervised attention."""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from SkinCancer.configs.model_config import BACKBONE_NAME, D_ATTN, IMG_SIZE, N_HEADS, N_SLOTS, SLOT_AGG
from SkinCancer.configs.training_config import (
    ACCUMULATION_STEPS, BATCH_SIZE, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY,
)
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.models.teme import smart_reshape
from SkinCancer.utils.seed import set_seed
from experiments.localize_sup import LTVSup, MaskSet
from experiments.residual import get_data, inv_freq, macro_auc, zero_last
from experiments.shared import Branch
import timm

OUT = os.path.join(OUTPUT_ROOT, "residual")
os.makedirs(OUT, exist_ok=True)

class HierSup(nn.Module):
    def __init__(self, pretrained=True, mode="bias"):
        super().__init__()
        self.net = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=7)
        if hasattr(self.net, "set_grad_checkpointing"):
            self.net.set_grad_checkpointing(True)
        with torch.no_grad():
            f, st = self.net.forward_intermediates(torch.randn(1, 3, IMG_SIZE, IMG_SIZE),
                                                   intermediates_only=False)
            s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
            self.s2_dim = s2.shape[1]
            final_dim = self.net.forward_head(f, pre_logits=True).shape[1]
        self.s1_head = nn.Linear(final_dim, 1)
        self.mel, self.nonmel = Branch(self.s2_dim, final_dim, 2), Branch(self.s2_dim, final_dim, 5)
        for br in (self.mel, self.nonmel):
            br.ltv_fusion = LTVSup(d_macro=final_dim, d_micro=256, d_attn=D_ATTN, n_heads=N_HEADS,
                                   n_slots=N_SLOTS, dropout=0.1, slot_agg=SLOT_AGG, mode=mode)
            zero_last(br)

    def branch(self, br, macro, s2):
        fused, sal = br.ltv_fusion(macro, br.micro_proj(smart_reshape(s2, self.s2_dim)))
        return br.classifier(fused), sal

    def forward(self, x):
        f, st = self.net.forward_intermediates(x, intermediates_only=False)
        s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
        macro = self.net.forward_head(f, pre_logits=True)
        flat = self.net.forward_head(f)
        hm, hs = macro.detach(), s2.detach()
        s1 = self.s1_head(hm).view(-1)
        g = torch.sigmoid(s1).view(-1, 1)
        cm, sm = self.branch(self.mel, hm, hs)
        cn, sn = self.branch(self.nonmel, hm, hs)
        return {"logits": flat + torch.cat([g * cm, (1 - g) * cn], 1), "flat": flat,
                "s1": s1, "sal": (sm, sn)}

def att_loss(sal, m):
    a = sal.mean(1).max(1).values.float()
    a = a / a.sum(1, keepdim=True).clamp_min(1e-12)
    t = m / m.sum(1, keepdim=True).clamp_min(1e-12)
    return -(t * torch.log(a.clamp_min(1e-12))).sum(1).mean()

@torch.no_grad()
def infer(model, loader):
    model.eval(); P, Pf, Y = [], [], []
    for img, lbl, _ in loader:
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            o = model(img.to(DEVICE))
        P.append(torch.softmax(o["logits"].float(), 1).cpu().numpy())
        Pf.append(torch.softmax(o["flat"].float(), 1).cpu().numpy())
        Y.append(np.asarray(lbl).flatten())
    return np.concatenate(P), np.concatenate(Pf), np.concatenate(Y).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=EPOCHS_STAGE1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    tag = f"h1sup_f{a.fold}_s{a.seed}"
    tr, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        tr = tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True)
    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    n_pos, n_neg = int((y <= 1).sum()), int((y > 1).sum())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE))
    model = HierSup(pretrained=not a.no_pretrained).to(DEVICE)
    bb = [p for n, p in model.named_parameters() if n.startswith("net.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("net.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": LEARNING_RATE},
                             {"params": hd, "lr": LEARNING_RATE * 10}],
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = DataLoader(MaskSet(tr, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(MaskSet(va, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[hier_sup:{tag}] lam={a.lam} train={len(tr)} val={len(va)}  "
          f"backbone+flat {sum(p.numel() for p in bb):,} / {sum(p.numel() for p in hd):,}  "
          f"lr {LEARNING_RATE:g} / {LEARNING_RATE*10:g}")

    ck, best = os.path.join(OUT, f"{tag}.pth"), -1.0
    for ep in range(a.epochs):
        model.train(); tc = ts = ta = 0.0
        opt.zero_grad(set_to_none=True)
        for i, (img, lbl, m) in enumerate(tl):
            img, lbl, m = img.to(DEVICE), lbl.to(DEVICE).long(), m.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                o = model(img)
                l_cls = ce(o["logits"], lbl)
                l_s1 = bce(o["s1"], (lbl <= 1).float())
                l_att = att_loss(o["sal"][0], m) + att_loss(o["sal"][1], m)
                loss = l_cls + l_s1 + a.lam * l_att
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tc += float(l_cls); ts += float(l_s1); ta += float(l_att)
        P, Pf, yv = infer(model, vl)
        auc = macro_auc(P, yv); n = max(len(tl), 1)
        print(f"Epoch {ep+1:02d}/{a.epochs} | cls {tc/n:.4f} s1 {ts/n:.4f} att {ta/n:.4f} | "
              f"Val AUC {auc:.4f} Acc {(P.argmax(1)==yv).mean():.4f} (flat {(Pf.argmax(1)==yv).mean():.4f})")
        if auc > best:
            best = auc; torch.save(model.state_dict(), ck)
    model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
    P, Pf, yv = infer(model, vl)
    f = os.path.join(OUT, f"h1sup_f{a.fold}_s{a.seed}_val_predictions.npz")
    np.savez(f, y_prob=P, y_prob_flat=Pf, y_true=yv, image_id=va["image_id"].values)
    print(f"[hier_sup:{tag}] val acc={(P.argmax(1)==yv).mean():.4f} (flat {(Pf.argmax(1)==yv).mean():.4f}) : {f}")

if __name__ == "__main__":
    main()
