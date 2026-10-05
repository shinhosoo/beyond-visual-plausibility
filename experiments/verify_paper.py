"""Recompute every number reported in the paper and compare it with the manuscript.

Reads only saved predictions and measurements, so no GPU and no training are needed.

  python -m experiments.verify_paper
"""
import os

import numpy as np
from scipy.stats import ttest_ind

from SkinCancer.configs.training_config import OUTPUT_ROOT

R = OUTPUT_ROOT
ROOT = os.path.dirname(os.path.abspath(R))
SEEDS = [42, 43, 44, 45, 46]
OK, BAD, MISS = 0, 0, 0


def check(label, got, want, tol=0.0015, unit=""):
    global OK, BAD, MISS
    if got is None:
        MISS += 1
        print(f"  {label:<46} {'(no data)':>22}   paper {want}{unit}")
        return
    hit = abs(got - want) <= tol
    OK, BAD = OK + hit, BAD + (not hit)
    mark = "ok" if hit else "MISMATCH"
    print(f"  {label:<46}{got:>12.4f}{unit}  paper {want}{unit}   {mark}")


def ld(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def route(d):
    p1, pm, pn = d["p_stage1"], d["p_mel"], d["p_nonmel"]
    o = np.zeros((len(p1), 7))
    o[:, 0] = p1 * (1 - pm); o[:, 1] = p1 * pm; o[:, 2:] = (1 - p1)[:, None] * pn
    return o


def f1_macro(y, p):
    v = []
    for c in range(7):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        v.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(v))


def flat_runs():
    out = []
    for s in SEEDS:
        f = os.path.join(R, "hier", ("flat" if s == 42 else f"flat_s{s}") + "_test_predictions.npz")
        d = ld(f)
        if d is None:
            continue
        y, p = d["y_true"].astype(int), d["y_prob"].argmax(1)
        out.append((float((p == y).mean()), f1_macro(y, p),
                    float((p[y == 1] == 1).mean()), float((p[y == 0] == 1).mean()),
                    float((p[y == 1] == 0).mean())))
    return out


def setting_runs(tag):
    out = []
    for s in SEEDS:
        if s == 42:
            k = {"baseline": "abl_baseline", "teme": "abl_teme", "ltv": "abl_ltv", "full": "abl_full"}[tag]
            d = ld(os.path.join(R, "experiments", f"{k}_test_predictions.npz")) or \
                ld(os.path.join(R, "experiments", f"{k}_test_predictions.npz"))
            if d is None:
                d2 = ld(os.path.join(ROOT, "runs", k, f"{k}_soft_predictions.npz"))
                if d2 is None:
                    continue
                y, p = d2["y_true"].astype(int), d2["y_prob"].argmax(1)
            else:
                y, p = d["y_true"].astype(int), route(d).argmax(1)
        else:
            d = ld(os.path.join(ROOT, f"outputs_seed{s}", "experiments", f"tbl_{tag}_s{s}_test_predictions.npz")) or \
                ld(os.path.join(ROOT, f"outputs_seed{s}", "experiments", f"tbl_{tag}_s{s}_test_predictions.npz"))
            if d is None:
                continue
            y, p = d["y_true"].astype(int), route(d).argmax(1)
        out.append((float((p == y).mean()), f1_macro(y, p),
                    float((p[y == 1] == 1).mean()), float((p[y == 0] == 1).mean()),
                    float((p[y == 1] == 0).mean())))
    return out


def mean_sd(v, i):
    a = np.array([x[i] for x in v])
    return (a.mean(), a.std(ddof=1)) if len(a) > 1 else (a.mean() if len(a) else None, None)


# ------------------------------------------------------------------ Table 1
def table1():
    print("\nTable 1  Ablation over five seeds")
    flat = flat_runs()
    print(f"  (runs found: single-stage {len(flat)})")
    if flat:
        m, s = mean_sd(flat, 0); check("single-stage accuracy", m, 0.8580); check("  s.d.", s, 0.0114)
        m, s = mean_sd(flat, 1); check("single-stage macro F1", m, 0.7383); check("  s.d.", s, 0.0229)
        m, s = mean_sd(flat, 2); check("single-stage MEL recall", m, 0.709, 0.004)
    want = {"baseline": (0.8435, 0.0125, 0.7062, 0.689),
            "ltv": (0.8456, 0.0114, 0.7077, 0.671),
            "teme": (0.8515, 0.0146, 0.7227, 0.669),
            "full": (0.8310, 0.0294, 0.6964, 0.692)}
    fa = np.array([x[0] for x in flat]) if flat else None
    for tag, lab in [("baseline", "2-stage soft"), ("ltv", "+LTV"), ("teme", "+TEME"), ("full", "+TEME+LTV")]:
        v = setting_runs(tag)
        if not v:
            print(f"  {lab:<46} (no data)")
            continue
        a, sd, f1, mel = want[tag]
        m, s = mean_sd(v, 0); check(f"{lab} accuracy  (n={len(v)})", m, a); check("  s.d.", s, sd)
        m, _ = mean_sd(v, 1); check(f"{lab} macro F1", m, f1)
        m, _ = mean_sd(v, 2); check(f"{lab} MEL recall", m, mel, 0.004)
        if fa is not None and len(v) > 1:
            p = ttest_ind(np.array([x[0] for x in v]), fa, equal_var=False).pvalue
            print(f"  {lab + ' p-value vs single-stage':<46}{p:>12.2f}")


# ------------------------------------------------------------------ Table 2
def table2():
    print("\nTable 2  MEL--NV confusion")
    for tag, lab, (r, mn, nm) in [
            (None, "1-stage baseline", (0.709, 0.197, 0.051)),
            ("baseline", "2-stage soft routing", (0.689, 0.207, 0.067)),
            ("full", "proposed (TEME+LTV)", (0.692, 0.206, 0.083))]:
        v = flat_runs() if tag is None else setting_runs(tag)
        if not v:
            print(f"  {lab:<46} (no data)"); continue
        check(f"{lab}: MEL recall", mean_sd(v, 2)[0], r, 0.004)
        check(f"{lab}: MEL->NV", mean_sd(v, 4)[0], mn, 0.004)
        check(f"{lab}: NV->MEL", mean_sd(v, 3)[0], nm, 0.004)


# ------------------------------------------------------------------ Table 3/4
def loc_metrics(tag, seeds=(42, 43, 44)):
    """Mean over folds and seeds of pointing / energy / IoU for one saliency map."""
    vals = {}
    for sd in seeds:
        base = os.path.join(R, "localize") if sd == 42 else os.path.join(R, "localize", f"seed{sd}")
        for k in range(5):
            d = ld(os.path.join(base, f"loc_{tag}_f{k}.npz"))
            if d is None:
                continue
            names = list(d["names"])
            for n in names:
                i = names.index(n)
                vals.setdefault(n, []).append(np.nanmean(d["metrics"][:, i, :], axis=0))
    return {n: np.mean(v, axis=0) for n, v in vals.items() if v}


def table3():
    print("\nTable 3  Explanation quality (overlap)")
    orig = loc_metrics("orig")
    sup = loc_metrics("sup0.2_bias_local")
    if orig:
        check("centred Gaussian: IoU", orig.get("center", [None] * 3)[2], 0.656, 0.004)
        check("Grad-CAM (stage-2): IoU", orig.get("gc2", [None] * 3)[2], 0.201, 0.004)
        check("LTV unsupervised: pointing", orig.get("ltv", [None] * 3)[0], 0.293, 0.004)
        check("LTV unsupervised: energy", orig.get("ltv", [None] * 3)[1], 0.293, 0.004)
        check("LTV unsupervised: IoU", orig.get("ltv", [None] * 3)[2], 0.208, 0.004)
    else:
        print("  (no overlap measurements found)")
    if sup:
        check("LTV + mask supervision: IoU", sup.get("ltv", [None] * 3)[2], 0.880, 0.006)
    print("\nTable 4  Masks needed")
    for frac, want in [("frac0.1", 0.856), ("frac0.25", 0.858), ("frac0.5", 0.882)]:
        m = loc_metrics(f"{frac}_bias_local", seeds=(42,))
        check(f"{frac}: IoU", m.get("ltv", [None] * 3)[2] if m else None, want, 0.006)


# ------------------------------------------------------------------ faithfulness
def table3_faith():
    print("\nTable 3  Explanation quality (faithfulness, insertion - deletion)")
    for tag, key, want in [("orig", "ltv", -0.013), ("orig", "gc2", 0.179),
                           ("orig", "center", 0.173), ("orig", "random", -0.001),
                           ("sup0.2_bias_local", "ltv", 0.178)]:
        vals = []
        for k in range(5):
            d = ld(os.path.join(R, "faithful", f"faith_{tag}_f{k}.npz"))
            if d is None:
                continue
            names = list(d["names"])
            if key not in names:
                continue
            i = names.index(key)
            vals.append(float(np.nanmean(d["ins"][:, i] - d["del"][:, i])))
        check(f"{tag} / {key}", float(np.mean(vals)) if vals else None, want, 0.01)


# ------------------------------------------------------------------ calibration
def calibration():
    print("\nStage-1 calibration")
    e, b = [], []
    for s in SEEDS:
        f = (os.path.join(R, "experiments", "abl_baseline_test_predictions.npz") if s == 42
             else os.path.join(ROOT, f"outputs_seed{s}", "experiments", f"tbl_baseline_s{s}_test_predictions.npz"))
        d = ld(f) or ld(f.replace("/experiments/", "/improve/"))
        if d is None:
            continue
        p1 = np.asarray(d["p_stage1"], float); y = d["y_true"].astype(int)
        mel = np.isin(y, [0, 1]).astype(int)
        idx = np.clip(np.digitize(p1, np.linspace(0, 1, 11)) - 1, 0, 9)
        ece = sum((idx == k).mean() * abs(mel[idx == k].mean() - p1[idx == k].mean())
                  for k in range(10) if (idx == k).sum())
        e.append(ece); b.append(float(np.mean((p1 - mel) ** 2)))
    check("ECE (mean over seeds)", float(np.mean(e)) if e else None, 0.0216, 0.001)
    check("Brier (mean over seeds)", float(np.mean(b)) if b else None, 0.0440, 0.001)


def main():
    print("=" * 84)
    print(" Verification of the reported numbers")
    print("=" * 84)
    table1(); table2(); table3(); table3_faith(); calibration()
    print("\n" + "=" * 84)
    print(f" matched {OK}   mismatched {BAD}   missing {MISS}")
    print("=" * 84)
    if BAD:
        print(" A mismatch means the manuscript and the saved predictions disagree; check the run.")
    if MISS:
        print(" Missing entries are experiments whose output files are not in this output root.")


if __name__ == "__main__":
    main()
