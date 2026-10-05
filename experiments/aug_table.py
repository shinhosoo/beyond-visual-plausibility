"""Print the augmentation comparison table."""
import glob
import os
import re

import numpy as np
from scipy.stats import binomtest

from SkinCancer.configs.training_config import OUTPUT_ROOT

D = os.path.join(OUTPUT_ROOT, "residual")
SETS = {"flat": ("flat", ["flatS", "flatM", "flatB"], "single-stage and full model"),
        "nm": ("nmnone", ["nmS", "nmM", "nmB"], "nonmel branch (1,549, )"),
        "mel": ("melnone", ["melS", "melM", "melB"], "mel branch")}


def oof(tag, seed=42):
    P, Y, I = [], [], []
    for k in range(5):
        f = os.path.join(D, f"{tag}_f{k}_s{seed}_val_predictions.npz")
        if not os.path.exists(f):
            return None, k
        d = np.load(f, allow_pickle=True)
        P.append(d["y_prob"]); Y.append(d["y_true"]); I.append(d["image_id"])
    I = np.concatenate(I); o = np.argsort(I)
    return (np.concatenate(P)[o], np.concatenate(Y)[o].astype(int), I[o]), 5


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main():
    print("=" * 92)
    print(" augmentation (cross-validation out-of-fold) S= M= mixup B=")
    print("=" * 92)
    for key, (base, variants, label) in SETS.items():
        rows = []
        b, nb = oof(base)
        for t in [base] + variants:
            r, n = oof(t)
            rows.append((t, r, n))
        if all(r is None for _, r, _ in rows):
            continue
        print(f"\n[{label}]")
        print(f"  {'settings':<10}{'fold':>7}{'n':>7}{'Acc':>9}{'macroF1':>10} ( McNemar)")
        for t, r, n in rows:
            if r is None:
                print(f"  {t:<10}{n:>5}/5 (progress )"); continue
            P, Y, I = r
            p = P.argmax(1); nc = P.shape[1]
            cmp = ""
            if t != base and b is not None:
                Pb, Yb, Ib = b
                pos = {x: j for j, x in enumerate(Ib)}
                if all(x in pos for x in I):
                    o = [pos[x] for x in I]; pb = Pb[o].argmax(1); yb = Yb[o]
                    n10 = int(((pb == yb) & (p != Y)).sum()); n01 = int(((pb != yb) & (p == Y)).sum())
                    pv = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
                    cmp = f" {n10:>4} / {t} {n01:>4}  p={pv:.3g}"
            print(f"  {t:<10}{'5/5':>7}{len(Y):>7}{(p==Y).mean():>9.4f}{macro_f1(Y,p,nc):>10.4f}   {cmp}")
    print("\n  The non-melanocytic branch is compared with the single-stage classifier.")
    print("  Augmentation settings are matched across configurations.")


if __name__ == "__main__":
    main()
