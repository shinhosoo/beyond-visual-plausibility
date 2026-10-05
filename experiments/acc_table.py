"""Print the accuracy table over seeds for every design setting."""
import os

import numpy as np
from scipy.stats import ttest_ind

from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics

ROOT = os.environ.get("HAM_OUTPUT_ROOT", "outputs")
F = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
SEEDS = [42, 43, 44, 45, 46]
RUNS = os.path.join(os.path.dirname(os.path.abspath(ROOT)), "runs")


def route(d, hard=False):
    p1, pm, pn = d["p_stage1"], d["p_mel"], d["p_nonmel"]
    g = (p1 > 0.5).astype(float) if hard else p1
    o = np.zeros((len(p1), 7)); o[:, 0] = g * (1 - pm); o[:, 1] = g * pm; o[:, 2:] = (1 - g)[:, None] * pn
    return o


def ld(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def get(setting, seed):
    imp = os.path.join(ROOT, "experiments")
    if setting == "1-stage":
        d = ld(os.path.join(ROOT, "hier", ("flat" if seed == 42 else f"flat_s{seed}") + "_test_predictions.npz"))
        return (d["y_prob"], d["y_true"]) if d is not None else None
    abl = {"2-stage soft": "baseline", "+LTV": "ltv", "+TEME": "teme", "+TEME+LTV": "full",
           "simple head": "simplehead", "2-stage hard": "baseline"}[setting]
    tbl = {"2-stage soft": "baseline", "+LTV": "ltv", "+TEME": "teme", "+TEME+LTV": "full",
           "simple head": "simple", "2-stage hard": "baseline"}[setting]
    hard = setting == "2-stage hard"
    if seed == 42:
        d = ld(os.path.join(RUNS, f"abl_{abl}", f"abl_{abl}_{'hard' if hard else 'soft'}_predictions.npz"))
        return (d["y_prob"], d["y_true"]) if d is not None else None
    cands = [os.path.join(imp, f"tbl_{tbl}_s{seed}_test_predictions.npz")]
    if setting == "+TEME+LTV":
        cands.insert(0, os.path.join(imp, f"seed{seed}_full_test_predictions.npz"))
    for c in cands:
        d = ld(c)
        if d is not None:
            return route(d, hard), d["y_true"]
    return None


def met(P, y):
    o, _ = calculate_detailed_multiclass_metrics(y.astype(int), P, F)
    p = P.argmax(1)
    return [o["Accuracy"], o["F1"], o["Sensitivity"], 100 * (p[y == 1] == 1).mean()]


def main():
    order = ["1-stage", "2-stage hard", "2-stage soft", "+LTV", "+TEME", "+TEME+LTV", "simple head"]
    print("=" * 96)
    print(" Accuracy over seeds on the held-out test split (mean +/- s.d.)")
    print("=" * 96)
    print(f"  {'settings':<14}{'seed':>12}{'Acc':>16}{'F1':>16}{'Sens':>16}{'MELrec':>14}{'vs 1-stage Acc':>18}")
    base = None
    for s in order:
        rows, got = [], []
        for sd in SEEDS:
            r = get(s, sd)
            if r is not None:
                rows.append(met(*r)); got.append(sd)
        if not rows:
            print(f"  {s:<14}{'-':>12}"); continue
        R = np.array(rows)
        sd_ = R.std(0, ddof=1) if len(R) > 1 else np.zeros(4)
        if s == "1-stage":
            base = R[:, 0]
        cmp = ""
        if base is not None and s != "1-stage" and len(R) > 1 and len(base) > 1:
            cmp = f"{100*(R[:,0].mean()-base.mean()):+.2f}%p p={ttest_ind(R[:,0], base, equal_var=False).pvalue:.2f}"
        print(f"  {s:<14}{','.join(map(str, got)):>12}"
              + "".join(f"{m:>9.4f}±{d:.4f}" for m, d in zip(R.mean(0)[:3], sd_[:3]))
              + f"{R[:,3].mean():>8.1f}±{sd_[3]:.1f}%{cmp:>18}")

    print("\n cross-validation ( settings seed training, fold val Acc)")
    for model in ["flat", "d3lr"]:
        line = []
        for sd in [42, 43, 44]:
            accs = []
            for k in range(5):
                d = ld(os.path.join(ROOT, "residual", f"{model}_f{k}_s{sd}_val_predictions.npz"))
                if d is not None:
                    accs.append((d["y_prob"].argmax(1) == d["y_true"]).mean())
            if len(accs) == 5:
                line.append((sd, np.array(accs)))
        if len(line) >= 2:
            diffs = np.concatenate([b - a for (_, a), (_, b) in zip(line[:-1], line[1:])])
            print(f"    {model:<5} seed {[s for s, _ in line]} Acc "
                  + " / ".join(f"{a.mean():.4f}" for _, a in line)
                  + f" fold seed || {np.abs(diffs).mean()*100:.2f}%p {np.abs(diffs).max()*100:.2f}%p")
        elif line:
            print(f"    {model:<5} seed {[s for s, _ in line]} done (2 )")

    print("\n ")
    print(" - each setting is compared with the single-stage classifier over the same seeds")
    print(" - at least three seeds are required for a t-test")


if __name__ == "__main__":
    main()
