"""Shared model and data helpers for the experiments."""
import argparse, os
import numpy as np, pandas as pd, timm, torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from SkinCancer.configs.data_config import FINAL7_TEST_CSV, SPLIT_ROOT
from SkinCancer.configs.model_config import BACKBONE_NAME, D_ATTN, IMG_SIZE, N_HEADS, N_SLOTS, SLOT_AGG
from SkinCancer.configs.training_config import ACCUMULATION_STEPS, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.models.ltv import LocalizeThenVerifyFusion
from SkinCancer.models.teme import LightweightTopoBlock, MultiScaleMicroEncoder, smart_reshape
from SkinCancer.utils.seed import set_seed

OUT = os.path.join(OUTPUT_ROOT, "shared"); os.makedirs(OUT, exist_ok=True)
FINAL7 = {"nv": 0, "mel": 1, "bkl": 2, "df": 3, "vasc": 4, "bcc": 5, "akiec": 6}


class Branch(nn.Module):
    """original Stage2Classifier backbone (TEME -> LTV -> )."""
    def __init__(self, s2_dim, final_dim, n_out, teme=True, ltv=True):
        super().__init__()
        self.s2_dim, self.ltv_on = s2_dim, ltv
        layers = [nn.Conv2d(s2_dim, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256),
                  nn.GELU(), MultiScaleMicroEncoder(256, 256)]
        if teme:
            layers.append(LightweightTopoBlock(256, reduced_dim=32))
        layers.append(nn.AdaptiveAvgPool2d(40))
        self.micro_proj = nn.Sequential(*layers)
        if ltv:
            self.ltv_fusion = LocalizeThenVerifyFusion(d_macro=final_dim, d_micro=256, d_attn=D_ATTN,
                n_heads=N_HEADS, n_slots=N_SLOTS, dropout=0.1, slot_agg=SLOT_AGG)
        else:
            self.macro_proj = nn.Linear(final_dim, D_ATTN)
            self.micro_vec = nn.Linear(256, D_ATTN)
            self.norm_out = nn.LayerNorm(D_ATTN)
        self.classifier = nn.Sequential(nn.Linear(D_ATTN, 128), nn.LayerNorm(128), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(128, n_out))

    def forward(self, macro, s2):
        micro = self.micro_proj(smart_reshape(s2, self.s2_dim))
        if self.ltv_on:
            fused, _ = self.ltv_fusion(macro, micro)
        else:
            fused = self.norm_out(self.macro_proj(macro) + self.micro_vec(micro.mean((2, 3))))
        return self.classifier(fused)


class SharedHier(nn.Module):
    def __init__(self, teme=True, ltv=True, pretrained=True):
        super().__init__()
        self.net = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=7)
        if hasattr(self.net, "set_grad_checkpointing"):
            self.net.set_grad_checkpointing(True)
        with torch.no_grad():
            f, st = self.net.forward_intermediates(torch.randn(1, 3, IMG_SIZE, IMG_SIZE), intermediates_only=False)
            s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
            s2_dim = s2.shape[1] if s2.dim() == 4 else s2.shape[-1]
            final_dim = self.net.forward_head(f, pre_logits=True).shape[1]
        self.s1_head = nn.Linear(final_dim, 1)
        self.mel = Branch(s2_dim, final_dim, 2, teme, ltv)
        self.nonmel = Branch(s2_dim, final_dim, 5, teme, ltv)

    def forward(self, x):
        f, st = self.net.forward_intermediates(x, intermediates_only=False)
        s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
        macro = self.net.forward_head(f, pre_logits=True)
        return {"flat": self.net.forward_head(f), "s1": self.s1_head(macro).view(-1),
                "mel": self.mel(macro, s2), "nonmel": self.nonmel(macro, s2)}


def compose(out):
    flat = torch.softmax(out["flat"].float(), 1)
    p1 = torch.sigmoid(out["s1"].float()).view(-1, 1)
    hier = torch.cat([p1 * torch.softmax(out["mel"].float(), 1),
                      (1 - p1) * torch.softmax(out["nonmel"].float(), 1)], 1)
    return {"hier": hier, "flat": flat, "avg": 0.5 * (hier + flat)}


def load_split(split, pm):
    csv = FINAL7_TEST_CSV if split == "test" else os.path.join(SPLIT_ROOT, f"global_{split}.csv")
    df = pd.read_csv(csv)
    if "diagnosis" not in df.columns and "dx" in df.columns:
        df = df.rename(columns={"dx": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].str.lower()
    df["image_path"] = df["image_id"].map(pm)
    df["label"] = df["diagnosis"].map(FINAL7)
    return df.dropna(subset=["image_path", "label"]).reset_index(drop=True)


def inv_freq(counts):
    counts = np.maximum(np.asarray(counts, float), 1)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=DEVICE)


def macro_auc(P, y):
    p = np.asarray(P, np.float64); p = p / p.sum(1, keepdims=True)
    try:
        return roc_auc_score(y, p, multi_class="ovr", average="macro")
    except Exception:
        return float((p.argmax(1) == y).mean())


@torch.no_grad()
def infer(model, loader):
    model.eval()
    acc = {"hier": [], "flat": [], "avg": [], "p1": [], "pm": [], "pn": []}; ys = []
    for img, lbl in tqdm(loader, leave=False):
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            out = model(img.to(DEVICE))
        c = compose(out)
        for k in ["hier", "flat", "avg"]:
            acc[k].append(c[k].cpu().numpy())
        acc["p1"].append(torch.sigmoid(out["s1"].float()).cpu().numpy())
        acc["pm"].append(torch.softmax(out["mel"].float(), 1).cpu().numpy())
        acc["pn"].append(torch.softmax(out["nonmel"].float(), 1).cpu().numpy())
        ys.append(lbl.numpy().flatten())
    return {k: np.concatenate(v) for k, v in acc.items()}, np.concatenate(ys).astype(int)


def train(a):
    set_seed(a.seed); tag = f"{a.tag}_s{a.seed}"
    pm = build_image_path_map()
    tr, va = load_split("train", pm), load_split("val", pm)
    if a.limit:
        tr = tr.groupby("label").head(max(2, a.limit // 7)).reset_index(drop=True)
        va = va.groupby("label").head(max(2, a.limit // 7)).reset_index(drop=True)
    y = tr["label"].astype(int).values
    w7 = inv_freq(np.bincount(y, minlength=7))
    n_pos, n_neg = int(np.isin(y, [0, 1]).sum()), int((~np.isin(y, [0, 1])).sum())
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    wm = inv_freq(np.bincount(y[np.isin(y, [0, 1])], minlength=2))
    wn = inv_freq(np.bincount(y[~np.isin(y, [0, 1])] - 2, minlength=5))
    ce7, bce = nn.CrossEntropyLoss(weight=w7), nn.BCEWithLogitsLoss(pos_weight=pw)
    cem, cen = nn.CrossEntropyLoss(weight=wm), nn.CrossEntropyLoss(weight=wn)
    print(f"[shared:{tag}] train={len(tr)} val={len(va)} TEME={a.teme} LTV={a.ltv} "
          f"lambda_s1={a.lam_s1} lambda_branch={a.lam_branch} pos_weight={pw.item():.4f}")
    model = SharedHier(a.teme, a.ltv, pretrained=not a.no_pretrained).to(DEVICE)
    print(f"[shared:{tag}] params={sum(p.numel() for p in model.parameters()):,}")
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = make_loader(tr, "label", build_train_transform(), shuffle=True)
    vl = make_loader(va, "label", build_val_transform(), shuffle=False)
    ckpt, best = os.path.join(OUT, f"{tag}.pth"), -1.0
    epochs = a.epochs or EPOCHS_STAGE1
    for ep in range(epochs):
        model.train()
        tot = {"all": 0.0, "flat": 0.0, "s1": 0.0, "mel": 0.0, "nonmel": 0.0}
        opt.zero_grad(set_to_none=True)
        for i, (img, lbl) in enumerate(tl):
            img, lbl = img.to(DEVICE), lbl.to(DEVICE).long()
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = model(img)
                l_flat = ce7(out["flat"], lbl)
                is_mel = lbl <= 1
                l_s1 = bce(out["s1"], is_mel.float())
                l_m = cem(out["mel"][is_mel], lbl[is_mel]) if is_mel.any() else out["mel"].sum() * 0
                l_n = cen(out["nonmel"][~is_mel], lbl[~is_mel] - 2) if (~is_mel).any() else out["nonmel"].sum() * 0
                loss = l_flat + a.lam_s1 * l_s1 + a.lam_branch * (l_m + l_n)
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            for k, v in [("all", loss), ("flat", l_flat), ("s1", l_s1), ("mel", l_m), ("nonmel", l_n)]:
                tot[k] += float(v)
        P, yv = infer(model, vl)
        auc = {k: macro_auc(P[k], yv) for k in ["hier", "flat", "avg"]}
        n = max(len(tl), 1)
        print(f"Epoch {ep+1:02d}/{epochs} | Loss {tot['all']/n:.4f} "
              f"(flat {tot['flat']/n:.3f} s1 {tot['s1']/n:.3f} mel {tot['mel']/n:.3f} nm {tot['nonmel']/n:.3f}) "
              f"| Val AUC hier {auc['hier']:.4f} flat {auc['flat']:.4f} avg {auc['avg']:.4f}")
        if auc[a.select] > best:
            best = auc[a.select]; torch.save(model.state_dict(), ckpt)
            print(f"  -> checkpoint saved (val {a.select} AUC {best:.4f})")
    print(f"[shared:{tag}] best val {a.select} AUC = {best:.4f} -> {ckpt}")


def predict(a):
    tag = f"{a.tag}_s{a.seed}"
    df = load_split(a.split, build_image_path_map())
    model = SharedHier(a.teme, a.ltv, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(OUT, f"{tag}.pth"), map_location=DEVICE, weights_only=True), strict=True)
    P, y = infer(model, make_loader(df, "label", build_val_transform(), shuffle=False))
    f = os.path.join(OUT, f"{tag}_{a.split}_predictions.npz")
    np.savez(f, y_true=y, image_id=df["image_id"].values, **P)
    print(f"[shared:{tag}] {a.split} n={len(y)} acc hier={(P['hier'].argmax(1)==y).mean():.4f} "
          f"flat={(P['flat'].argmax(1)==y).mean():.4f} avg={(P['avg'].argmax(1)==y).mean():.4f} : {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "predict"])
    ap.add_argument("--tag", default="A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--teme", type=int, default=1)
    ap.add_argument("--ltv", type=int, default=1)
    ap.add_argument("--lam-s1", type=float, default=1.0)
    ap.add_argument("--lam-branch", type=float, default=1.0)
    ap.add_argument("--select", default="hier", choices=["hier", "flat", "avg"])
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    (train if a.stage == "train" else predict)(a)


if __name__ == "__main__":
    main()
