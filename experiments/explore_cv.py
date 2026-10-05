"""Cross-validated version of the explore experiments."""
import argparse, os
import numpy as np
from scipy.stats import binomtest
from SkinCancer.configs.training_config import OUTPUT_ROOT
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics

F = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
NOISE = 0.006


def load(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True); o = np.argsort(d["image_id"])
    return d["y_prob"][o], d["y_true"][o].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    base = lambda k: load(os.path.join(OUTPUT_ROOT, "residual", f"flat_f{k}_s{a.seed}_val_predictions.npz"))
    print("=" * 92); print(f" (baseline = cross-validation flat, ±{NOISE*100:.1f}%p)"); print("=" * 92)
    for n in a.names:
        rows, Y, A, B = [], [], [], []
        for k in range(5):
            fb, fm = base(k), load(os.path.join(OUTPUT_ROOT, "explore", f"{n}_f{k}_s{a.seed}_val_predictions.npz"))
            if fb is None or fm is None:
                continue
            (pb, y), (pm, _) = fb, fm
            ob, _ = calculate_detailed_multiclass_metrics(y, pb, F); om, _ = calculate_detailed_multiclass_metrics(y, pm, F)
            mb = 100 * (pb.argmax(1)[y == 1] == 1).mean(); mm = 100 * (pm.argmax(1)[y == 1] == 1).mean()
            rows.append((k, om["Accuracy"] - ob["Accuracy"], om["F1"] - ob["F1"], mm - mb))
            Y.append(y); A.append(pb.argmax(1)); B.append(pm.argmax(1))
        if not rows:
            print(f"\n[{n}] comparison fold missing"); continue
        print(f"\n[{n}]")
        for k, da, df, dm in rows:
            flag = "  +" if da > NOISE else ("  -" if da < -NOISE else "")
            print(f"  fold {k}:  ΔAcc {da*100:+.2f}%p   ΔF1 {df*100:+.2f}%p   ΔMELrec {dm:+.1f}%p{flag}")
        y, pa, pb_ = np.concatenate(Y), np.concatenate(A), np.concatenate(B)
        n10 = int(((pa == y) & (pb_ != y)).sum()); n01 = int(((pa != y) & (pb_ == y)).sum())
        p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
        up = sum(r[1] > NOISE for r in rows)
        ok = up >= int(np.ceil(len(rows) * 2 / 3)) and n01 > n10
        print(f" n={len(y)}: flat {n10} / {n} {n01}  p={p:.3g}")
        print(f" fold {up}/{len(rows)}  ->  {' (five folds)' if ok else ''}")


if __name__ == "__main__":
    main()
