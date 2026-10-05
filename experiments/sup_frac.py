"""How many masks does the attention supervision need?

Same procedure as localize_sup, except that the alignment term is applied to a fixed subset of
the training images, selected by a hash of the image identifier so that the subset is identical
in every fold and epoch. All images still contribute the classification loss.

  python -m experiments.sup_frac --stage train --fold 0 --frac 0.25 --lam 0.2
  python -m experiments.sup_frac --stage eval  --fold 0 --frac 0.25
"""
import argparse
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import experiments.localize as L
import experiments.localize_sup as S
from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.models.teme import smart_reshape
from SkinCancer.utils.seed import set_seed
from experiments.residual import get_data, inv_freq

class MaskSetFrac(S.MaskSet):
    """returns (x, y, m, keep); only samples with keep=1 enter the attention loss"""

    def __init__(self, df, train, frac, grid=40):
        super().__init__(df, train, grid)
        self.keep = [int(hashlib.md5(str(i).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac
                     for i in self.ids]

    def __getitem__(self, i):
        x, y, m = super().__getitem__(i)
        return x, y, m, float(self.keep[i])

def train(a, tag):
    set_seed(a.seed)
    tr, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        tr = tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True)
    net = L.load_flat(a.fold)
    br, s2_dim = S.build(net)
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(tr["label"].astype(int).values, minlength=7)))
    opt = torch.optim.AdamW(br.parameters(), lr=5e-4, weight_decay=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    ds = MaskSetFrac(tr, True, a.frac)
    tl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(S.MaskSet(va, False), batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)
    n_sup = int(sum(ds.keep))
    print(f"[frac:{tag}] frac={a.frac} lam={a.lam} train={len(tr)} "
          f"(mask {n_sup}, {100*n_sup/max(len(tr),1):.1f}%) val={len(va)}")

    best, ck = -1.0, os.path.join(L.OUT, f"readout_{tag}_f{a.fold}.pth")
    for ep in range(a.epochs):
        br.train(); tc = ta = 0.0; used = 0
        for img, lbl, m, keep in tl:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE).long()
            m, keep = m.to(DEVICE), keep.to(DEVICE).float()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                f, s2, _ = L.feats(net, img)
                macro = net.forward_head(f, pre_logits=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                micro = br.micro_proj(smart_reshape(s2.float(), s2_dim))
                fused, sal = br.ltv_fusion(macro.float(), micro)
                z = br.classifier(fused)
                att = sal.mean(1).max(1).values.float()
                att = att / att.sum(1, keepdim=True).clamp_min(1e-12)
                tgt = m / m.sum(1, keepdim=True).clamp_min(1e-12)
                per = -(tgt * torch.log(att.clamp_min(1e-12))).sum(1)
                l_att = (per * keep).sum() / keep.sum().clamp_min(1.0)
                l_cls = ce(z, lbl)
                loss = l_cls + a.lam * l_att
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tc += float(l_cls); ta += float(l_att); used += int(keep.sum())
        br.eval(); P, Y = [], []
        with torch.no_grad():
            for img, lbl, _ in vl:
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    f, s2, _ = L.feats(net, img.to(DEVICE))
                    z, _ = L.branch_forward(br, s2_dim,
                                            net.forward_head(f, pre_logits=True).float(), s2.float())
                P.append(torch.softmax(z.float(), 1).cpu().numpy()); Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr", average="macro")
        except Exception:
            auc = float((P.argmax(1) == Y).mean())
        n = max(len(tl), 1)
        print(f"Epoch {ep+1:02d}/{a.epochs} | cls {tc/n:.4f} att {ta/n:.4f} (supervised {used}) "
              f"| readout Val AUC {auc:.4f} Acc {(P.argmax(1)==Y).mean():.4f}")
        if auc > best:
            best = auc; torch.save(br.state_dict(), ck)
    print(f"[frac:{tag}] best readout val AUC {best:.4f} -> {ck}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tag = f"frac{a.frac:g}_bias_local"
    if a.seed != 42:
        L.OUT = os.path.join(L.OUT, f"seed{a.seed}"); os.makedirs(L.OUT, exist_ok=True)
    if a.stage == "train":
        train(a, tag)
    else:
        L.make_branch = lambda net, _m: S.build(net)
        a.ltv_mode = tag
        L.evaluate(a)

if __name__ == "__main__":
    main()
