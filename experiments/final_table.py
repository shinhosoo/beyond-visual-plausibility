"""Print the final table of the configurations evaluated on the test split."""
import os

import numpy as np
from scipy.stats import ttest_ind

from SkinCancer.configs.training_config import OUTPUT_ROOT
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics

F = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
SEEDS = [42, 43, 44, 45, 46]


def ld(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def get(model, s):
    if model == "1-stage":
        d = ld(os.path.join(OUTPUT_ROOT, "hier", ("flat" if s == 42 else f"flat_s{s}") + "_test_predictions.npz"))
        return (d["y_prob"], d["y_true"]) if d is not None else None
    d = ld(os.path.join(OUTPUT_ROOT, "residual", f"{model}_full_s{s}_test_predictions.npz"))
    return (d["y_prob"], d["y_true"]) if d is not None else None


def met(P, y):
    o, _ = calculate_detailed_multiclass_metrics(y.astype(int), P, F)
    p = P.argmax(1)
    return [o["Accuracy"], o["F1"], o["Sensitivity"], 100 * (p[y == 1] == 1).mean()]


def main():
    print("=" * 92)
    print(" evaluation comparison (table split, settings) = ± s.d.")
    print("=" * 92)
    print(f"  {'':<16}{'seed':>12}{'Acc':>16}{'F1':>16}{'Sens':>14}{'MELrec':>12}{'vs 1-stage':>16}")
    base = None
    for m, label in [("1-stage", "1-stage"), ("d3lr", "correction (d3lr)"), ("d3", "correction(default lr)"),
                     ("d1lr", " correction")]:
        rows, got = [], []
        for s in SEEDS:
            r = get(m, s)
            if r is not None:
                rows.append(met(*r)); got.append(s)
        if not rows:
            print(f"  {label:<16}{'-':>12}"); continue
        R = np.array(rows); sd = R.std(0, ddof=1) if len(R) > 1 else np.zeros(4)
        if m == "1-stage":
            base = R[:, 0]
        cmp = ""
        if base is not None and m != "1-stage" and len(R) > 1:
            cmp = f"{100*(R[:,0].mean()-base.mean()):+.2f}%p p={ttest_ind(R[:,0], base, equal_var=False).pvalue:.2f}"
        print(f"  {label:<16}{','.join(map(str,got)):>12}"
              + "".join(f"{a:>9.4f}±{b:.4f}" for a, b in zip(R.mean(0)[:2], sd[:2]))
              + f"{R[:,2].mean():>9.4f}±{sd[2]:.3f}{R[:,3].mean():>10.1f}%{cmp:>16}")
    print("\n  Differences from the single-stage classifier with p > 0.05 are within the seed-to-seed spread.")


if __name__ == "__main__":
    main()
