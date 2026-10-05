"""Print the table for the magnified re-input experiment."""
import glob
import os

import numpy as np
from scipy.stats import binomtest

from SkinCancer.configs.training_config import OUTPUT_ROOT

D = os.path.join(OUTPUT_ROOT, "zoom")
COND = ["base", "rand", "center", "maskbox", "ltvbox", "ltvonly"]
LABEL = {"base": "full (baseline)", "rand": "+ ( TTA)",
         "center": "+ ( )", "maskbox": "lesion box",
         "ltvbox": "LTV box", "ltvonly": "LTV only"}


def macro_f1(y, p, n=7):
    f = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def main():
    fs = sorted(glob.glob(os.path.join(D, "zoom_f*_s*.npz")))
    if not fs:
        print(" results missing"); return
    parts = [np.load(f, allow_pickle=True) for f in fs]
    y = np.concatenate([p["y_true"] for p in parts]).astype(int)
    P = {c: np.concatenate([p[c] for p in parts]) for c in COND}
    print("=" * 88)
    print(f" ( {len(fs)}, n={len(y)}, 1 )")
    print("=" * 88)
    print(f"  {'':<26}{'Acc':>9}{'macroF1':>10}{'vs base':>10} McNemar")
    b = P["base"].argmax(1)
    for c in COND:
        p = P[c].argmax(1)
        cmp = ""
        if c != "base":
            n10 = int(((b == y) & (p != y)).sum()); n01 = int(((b != y) & (p == y)).sum())
            pv = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
            cmp = f"base {n10:>4} / {c} {n01:>4}  p={pv:.3g}"
        print(f"  {LABEL[c]:<26}{(p==y).mean():>9.4f}{macro_f1(y,p):>10.4f}"
              f"{(p==y).mean()-(b==y).mean():>+10.4f}   {cmp}")
    print("\n ")
    print(" - ltvbox against random and centre crops isolates the lesion location")
    print(" - ltvbox close to maskbox means the LTV box approximates the lesion mask")
    print(" - ltvonly removes the surrounding context entirely")


if __name__ == "__main__":
    main()
