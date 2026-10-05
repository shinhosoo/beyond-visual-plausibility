"""Supervise the LTV attention with the lesion masks.

The alignment term is a cross-entropy between the attention map and the pooled mask. Masks are
used during training only; inference needs none.

  python -m experiments.localize_sup --stage train --fold 0 --lam 0.2
  python -m experiments.localize_sup --stage eval  --fold 0 --lam 0.2
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

import experiments.localize as L
from SkinCancer.configs.model_config import D_ATTN, IMG_SIZE, N_HEADS, N_SLOTS, SLOT_AGG
from SkinCancer.configs.training_config import BATCH_SIZE, DEVICE, OUTPUT_ROOT
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.models.teme import smart_reshape
from SkinCancer.utils.seed import set_seed
from experiments.ltv_fix import LTVFixed
from experiments.localize_x import Zero
from experiments.residual import get_data, inv_freq
from experiments.shared import Branch

MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
JIT = transforms.ColorJitter(brightness=0.1, contrast=0.1)


class LTVSup(LTVFixed):
    """LTVFixed saliency detach (attention supervised)."""

    def forward(self, macro_feat, gcn_feat):
        B, C, H, W = gcn_feat.shape
        N, h, hd = H * W, self.n_heads, self.head_dim
        tokens = gcn_feat.flatten(2).transpose(1, 2)
        slots = self.slot_init(macro_feat).view(B, self.n_slots, self.d_attn)
        Q1 = self.p1_q_proj(self.norm1_q(slots)).view(B, self.n_slots, h, hd).transpose(1, 2)
        K1 = self.p1_k_proj(self.norm1_kv(tokens)).view(B, N, h, hd).transpose(1, 2)
        sal = ((Q1 @ K1.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        imp = sal.mean(dim=1).max(dim=1).values
        w = imp / imp.max(dim=1, keepdim=True).values.clamp_min(1e-12)
        kv2 = self.norm2_kv(tokens)
        if self.mode == "norm":
            kv2 = kv2 * w.unsqueeze(-1)
        Q2 = self.p2_q_proj(self.norm2_q(slots)).view(B, self.n_slots, h, hd).transpose(1, 2)
        K2 = self.p2_k_proj(kv2).view(B, N, h, hd).transpose(1, 2)
        V2 = self.p2_v_proj(kv2).view(B, N, h, hd).transpose(1, 2)
        lg = (Q2 @ K2.transpose(-2, -1)) * self.scale
        if self.mode == "bias":
            lg = lg + torch.log(w.clamp_min(1e-6)).to(lg.dtype)[:, None, None, :]
        a = self.dropout(lg.softmax(dim=-1))
        out = (a @ V2).transpose(1, 2).contiguous().view(B, self.n_slots, self.d_attn)
        slots = slots + self.p2_out(out)
        slots = slots + self.p2_ffn(slots)
        if self.slot_agg == "mean":
            agg = slots.mean(dim=1)
        elif self.slot_agg == "max":
            agg = slots.max(dim=1).values
        else:
            agg = (torch.softmax(self.slot_weight_proj(macro_feat), dim=-1).unsqueeze(-1) * slots).sum(dim=1)
        return self.norm_out(self.out_proj(agg) + self.macro_proj(macro_feat)), sal


class MaskSet(Dataset):
    """ + + 40x40 lesion mask ( augmentation )."""

    def __init__(self, df, train, grid=40):
        self.p = df["image_path"].tolist(); self.y = df["label"].astype(int).tolist()
        self.ids = df["image_id"].tolist(); self.train = train; self.g = grid

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        img = Image.open(self.p[i]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mp = os.path.join(MASK_DIR, f"{self.ids[i]}_segmentation.png")
        msk = Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        if self.train:
            if random.random() < 0.5:
                img, msk = TF.hflip(img), TF.hflip(msk)
            if random.random() < 0.2:
                img, msk = TF.vflip(img), TF.vflip(msk)
            ang = random.uniform(-30, 30)
            img = TF.rotate(img, ang, interpolation=TF.InterpolationMode.BILINEAR)
            msk = TF.rotate(msk, ang, interpolation=TF.InterpolationMode.NEAREST)
            img = JIT(img)
        x = NORM(TF.to_tensor(img))
        m = torch.tensor(np.array(msk) > 127, dtype=torch.float32)[None, None]
        m = Fn.adaptive_avg_pool2d(m, self.g)[0, 0].reshape(-1)
        return x, self.y[i], m


def build(net, mode="bias", local=True):
    with torch.no_grad():
        f, s2, _ = L.feats(net, torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
        s2_dim = s2.shape[1]; final_dim = net.forward_head(f, pre_logits=True).shape[1]
    br = Branch(s2_dim, final_dim, 7, teme=True, ltv=True)
    br.ltv_fusion = LTVSup(d_macro=final_dim, d_micro=256, d_attn=D_ATTN, n_heads=N_HEADS,
                           n_slots=N_SLOTS, dropout=0.1, slot_agg=SLOT_AGG, mode=mode)
    if local:
        br.ltv_fusion.macro_proj = Zero(D_ATTN)
    return br.to(DEVICE), s2_dim


def train(a, tag):
    set_seed(a.seed)
    tr, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        tr = tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True)
    net = L.load_flat(a.fold)
    br, s2_dim = build(net)
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(tr["label"].astype(int).values, minlength=7)))
    opt = torch.optim.AdamW(br.parameters(), lr=5e-4, weight_decay=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = DataLoader(MaskSet(tr, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(MaskSet(va, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[sup:{tag}] lam={a.lam} train={len(tr)} val={len(va)} params={sum(p.numel() for p in br.parameters()):,}")
    best, ck = -1.0, os.path.join(L.OUT, f"readout_{tag}_f{a.fold}.pth")
    for ep in range(a.epochs):
        br.train(); tc = ta = 0.0
        for img, lbl, m in tl:
            img, lbl, m = img.to(DEVICE), lbl.to(DEVICE).long(), m.to(DEVICE)
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
                l_att = -(tgt * torch.log(att.clamp_min(1e-12))).sum(1).mean()
                l_cls = ce(z, lbl)
                loss = l_cls + a.lam * l_att
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tc += float(l_cls); ta += float(l_att)
        br.eval(); P, Y = [], []
        with torch.no_grad():
            for img, lbl, _ in vl:
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    f, s2, _ = L.feats(net, img.to(DEVICE))
                    z, _ = L.branch_forward(br, s2_dim, net.forward_head(f, pre_logits=True).float(), s2.float())
                P.append(torch.softmax(z.float(), 1).cpu().numpy()); Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr", average="macro")
        except Exception:
            auc = float((P.argmax(1) == Y).mean())
        n = max(len(tl), 1)
        print(f"Epoch {ep+1:02d}/{a.epochs} | cls {tc/n:.4f} att {ta/n:.4f} | readout Val AUC {auc:.4f} "
              f"Acc {(P.argmax(1)==Y).mean():.4f}")
        if auc > best:
            best = auc; torch.save(br.state_dict(), ck)
    print(f"[sup:{tag}] best readout val AUC {best:.4f} -> {ck}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tag = f"sup{a.lam:g}_bias_local"
    if a.seed != 42:
        L.OUT = os.path.join(L.OUT, f"seed{a.seed}"); os.makedirs(L.OUT, exist_ok=True)
    if a.stage == "train":
        train(a, tag)
    else:
        L.make_branch = lambda net, _m: build(net)
        a.ltv_mode = tag
        L.evaluate(a)


if __name__ == "__main__":
    main()
