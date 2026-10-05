"""Summarise the overlap measurements across configurations and folds."""
import argparse
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon

from SkinCancer.configs.training_config import OUTPUT_ROOT

OUT = os.path.join(OUTPUT_ROOT, "localize")
F = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]
MET = ["pointing", "energy", "IoU"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="orig")
    a = ap.parse_args()
    parts, done = [], []
    for k in range(5):
        f = os.path.join(OUT, f"loc_{a.mode}_f{k}.npz")
        if os.path.exists(f):
            parts.append(np.load(f, allow_pickle=True)); done.append(k)
    if not parts:
        print("results missing"); return
    R = np.concatenate([p["metrics"] for p in parts])
    names = list(parts[0]["names"])
    lab = np.concatenate([p["label"] for p in parts])
    area = np.concatenate([p["lesion_area"] for p in parts])
    fp = np.concatenate([p["flat_pred"] for p in parts])
    rp = np.concatenate([p["readout_pred"] for p in parts])
    L = names.index("ltv")
    print("=" * 84)
    print(f" mode={a.mode} fold {done}  n={len(lab)} (test )")
    print(f" 1-stage accuracy {np.mean(fp==lab):.4f} ( ) LTV readout accuracy {np.mean(rp==lab):.4f} ()")
    print("=" * 84)
    print(f"  {'supervised':<8}" + "".join(f"{m:>12}" for m in MET))
    for j, n in enumerate(names):
        print(f"  {n:<8}" + "".join(f"{np.nanmean(R[:, j, i]):>12.3f}" for i in range(3)))
    print("\nLTV attention versus the reference maps")
    for j, n in enumerate(names):
        if j == L:
            continue
        a_, b_ = R[:, L, 0].astype(bool), R[:, j, 0].astype(bool)
        n10, n01 = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
        pp = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
        msk = ~np.isnan(R[:, L, 1]) & ~np.isnan(R[:, j, 1])
        de = R[msk, L, 1] - R[msk, j, 1]
        pe = wilcoxon(R[msk, L, 1], R[msk, j, 1]).pvalue
        mi = ~np.isnan(R[:, L, 2]) & ~np.isnan(R[:, j, 2])
        di = R[mi, L, 2] - R[mi, j, 2]
        pi = wilcoxon(R[mi, L, 2], R[mi, j, 2]).pvalue
        print(f"    vs {n:<7} pointing LTV {n10:>4} / {n} {n01:>4} p={pp:.2g}   "
              f"energy Δ{np.mean(de):+.3f} p={pe:.2g}   IoU Δ{np.mean(di):+.3f} p={pi:.2g}")
    ok = []
    for tgt in ("gc4", "center"):
        j = names.index(tgt)
        msk = ~np.isnan(R[:, L, 1]) & ~np.isnan(R[:, j, 1])
        d = np.mean(R[msk, L, 1] - R[msk, j, 1]); p = wilcoxon(R[msk, L, 1], R[msk, j, 1]).pvalue
        ok.append(d > 0 and p < 0.05)
    print(f"\n (energy gc4 center ): {'' if all(ok) else ''}")
    print("\n  Energy ratio inside the lesion mask")
    qs = np.quantile(area, [0, 1/3, 2/3, 1])
    for lo, hi, nm in [(qs[0], qs[1], ""), (qs[1], qs[2], ""), (qs[2], qs[3] + 1e-9, "")]:
        m = (area >= lo) & (area < hi)
        if not m.any():
            continue
        print(f"    {nm} ( {lo:.2f}~{hi:.2f}, n={m.sum()}): "
              + "  ".join(f"{n} {np.nanmean(R[m, j, 1]):.3f}" for j, n in enumerate(names)))
    print("\n energy")
    for c in range(7):
        m = lab == c
        if m.sum():
            print(f"    {F[c]:<6} n={m.sum():>5}  " + "  ".join(f"{n} {np.nanmean(R[m, j, 1]):.3f}" for j, n in enumerate(names)))


if __name__ == "__main__":
    main()
