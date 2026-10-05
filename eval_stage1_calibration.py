"""Expected calibration error and Brier score of the Stage 1 routing probabilities."""
import argparse, glob, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from SkinCancer.configs.data_config import FINAL7_TEST_CSV, STAGE1_MAPPING, STAGE1_VAL_CSV
from SkinCancer.configs.model_config import BACKBONE_NAME
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.dataset import LocalSkinCancerDataset
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.models.stage1 import create_stage1_classifier

PAPER_ECE, PAPER_BRIER = 0.0295, 0.0430
TOL_ECE, TOL_BRIER = 0.005, 0.003

def load_split(csv_path, path_map):
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns and "dx" in df.columns:
        df = df.rename(columns={"dx": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].str.lower()
    df["label"] = df["diagnosis"].map(STAGE1_MAPPING)
    df["image_path"] = df["image_id"].map(path_map)
    return df.dropna(subset=["label", "image_path"]).reset_index(drop=True)

@torch.no_grad()
def predict(model, df, bs=32):
    ds = LocalSkinCancerDataset(df["image_path"].tolist(), df["label"].astype(float).tolist(), build_val_transform())
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    model.eval(); probs, labels = [], []
    for img, lbl in loader:
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            out = model(img.to(DEVICE))
        probs.append(torch.sigmoid(out).float().cpu().numpy().ravel()); labels.append(lbl.numpy().ravel())
    return np.concatenate(probs), np.concatenate(labels).astype(int)

def calibration(p, y, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1); ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0: rows.append((lo, hi, 0, np.nan, np.nan)); continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += (n / len(p)) * abs(acc - conf); rows.append((lo, hi, n, conf, acc))
    return float(ece), float(np.mean((p - y) ** 2)), rows

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None); ap.add_argument("--tag", default=None)
ap.add_argument("--batch-size", type=int, default=32)
args = ap.parse_args()

ckpt = args.ckpt
if ckpt is None:
    c = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "stage1_*.pth")), key=os.path.getmtime)
    if not c: sys.exit("checkpoint missing")
    ckpt = c[-1]
tag = args.tag or f"wd{os.environ.get('HAM_WEIGHT_DECAY','0.05')}_aug{os.environ.get('HAM_AUGMENT','code')}"
print(f"checkpoint : {ckpt}\ntag        : {tag}\n")

pm = build_image_path_map()
te, va = load_split(FINAL7_TEST_CSV, pm), load_split(STAGE1_VAL_CSV, pm)
print(f"test={len(te)}  val={len(va)}\n")
model = create_stage1_classifier(backbone_name=BACKBONE_NAME, pretrained=False).to(DEVICE)
model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=True)

p_te, y_te = predict(model, te, args.batch_size)
p_va, y_va = predict(model, va, args.batch_size)
ece, brier, rows = calibration(p_te, y_te)
auc_te = roc_auc_score(y_te, p_te)
fpr, tpr, thr = roc_curve(y_va, p_va); youden = float(thr[np.argmax(tpr - fpr)])

ok_e, ok_b = abs(ece-PAPER_ECE) <= TOL_ECE, abs(brier-PAPER_BRIER) <= TOL_BRIER
print("="*62); print("  Stage 1 calibration on test split  (paper Table 6)"); print("="*62)
print(f"  ECE (10 bins)   {ece:.4f} {PAPER_ECE:.4f} {ece-PAPER_ECE:+.4f}   {'OK' if ok_e else ''}")
print(f"  Brier score     {brier:.4f} {PAPER_BRIER:.4f} {brier-PAPER_BRIER:+.4f}   {'OK' if ok_b else ''}")
print(f"  test AUC        {auc_te:.4f} ( )")
print(f" Youden {youden:.4f} (val )")
print(f"\n  {'bin':<14}{'n':>6}{'conf':>10}{'acc':>10}{'|acc-conf|':>12}")
for lo, hi, n, conf, acc in rows:
    if n == 0: print(f"  [{lo:.1f}, {hi:.1f}]{'':<4}{0:>6}{'-':>10}{'-':>10}{'-':>12}")
    else: print(f"  [{lo:.1f}, {hi:.1f}]{'':<4}{n:>6}{conf:>10.4f}{acc:>10.4f}{abs(acc-conf):>12.4f}")
print("\n"+"="*62)
print("  Calibration is reported for the Stage 1 router only." if (ok_e and ok_b) else " : - ")
print("="*62)

out = os.path.join(OUTPUT_ROOT, "stage1_calibration.csv")
row = pd.DataFrame([{"tag": tag, "checkpoint": os.path.basename(ckpt),
    "weight_decay": os.environ.get("HAM_WEIGHT_DECAY","0.05"), "augment": os.environ.get("HAM_AUGMENT","code"),
    "n_test": len(y_te), "ECE": round(ece,6), "Brier": round(brier,6), "test_AUC": round(auc_te,6),
    "youden_threshold": round(youden,6), "match": bool(ok_e and ok_b)}])
if os.path.exists(out): row = pd.concat([pd.read_csv(out), row], ignore_index=True)
row.to_csv(out, index=False)
np.savez(os.path.join(OUTPUT_ROOT, f"stage1_probs_{tag}.npz"),
         p_test=p_te, y_test=y_te, p_val=p_va, y_val=y_va, image_id=te["image_id"].values)
print(f"\n: {out}")
