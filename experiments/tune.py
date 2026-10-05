"""Threshold sweep for the melanocytic branch, selected on the validation split."""

import argparse
import glob
import os

import numpy as np
from scipy.stats import binomtest

from SkinCancer.configs.data_config import FINAL_CLASS_NAMES
from SkinCancer.configs.training_config import OUTPUT_ROOT
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics

EXPERIMENTS_DIR = os.path.join(OUTPUT_ROOT, "experiments")
F = FINAL_CLASS_NAMES


def route(p1, pm, pn, thr=0.5):
    """Apply a threshold to the melanocytic branch and rebuild the seven-class distribution.

  p_adj > 0.5 is equivalent to p_mel > thr; thr = 0.5 reproduces the default routing.
"""
    eps = 1e-7
    if abs(thr - 0.5) > eps:
        lp = np.log(np.clip(pm, eps, 1 - eps) / np.clip(1 - pm, eps, 1 - eps))
        lt = np.log(thr / (1 - thr))
        pm = 1.0 / (1.0 + np.exp(-(lp - lt)))
    out = np.zeros((len(p1), 7), dtype=np.float64)
    out[:, 0] = p1 * (1 - pm)
    out[:, 1] = p1 * pm
    out[:, 2:7] = (1 - p1)[:, None] * pn
    return out


def load(tag, split):
    f = os.path.join(EXPERIMENTS_DIR, f"{tag}_{split}_predictions.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return d["p_stage1"], d["p_mel"], d["p_nonmel"], d["y_true"].astype(int), d["image_id"]


def macro(y, P, key="F1"):
    o, _ = calculate_detailed_multiclass_metrics(y, P, F)
    return o[key], o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--select-on", default="F1",
                    choices=["F1", "Accuracy", "Sensitivity", "AUC"],
                    help="val table")
    ap.add_argument("--baseline", default=None,
                    help=" comparison tag (test prediction )")
    args = ap.parse_args()

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    results = {}

    for tag in args.tags:
        v = load(tag, "val")
        t = load(tag, "test")
        if v is None or t is None:
            print(f"[{tag}] prediction missing (val={v is not None}, test={t is not None}) - ")
            continue
        p1v, pmv, pnv, yv, _ = v
        p1t, pmt, pnt, yt, idt = t

        print(f"\n{'='*70}\n[{tag}]  val n={len(yv)}  test n={len(yt)}\n{'='*70}")
        print(f" val ( : macro {args.select_on})")
        print(f"  {'thr':>6}{'val '+args.select_on:>12}{'MEL prediction':>12}")
        best, best_thr = -1.0, 0.5
        for thr in grid:
            Pv = route(p1v, pmv, pnv, thr)
            s, _ = macro(yv, Pv, args.select_on)
            nmel = int((Pv.argmax(1) == 1).sum())
            mark = ""
            if s > best:
                best, best_thr, mark = s, thr, "  <-"
            print(f"  {thr:>6.2f}{s:>12.4f}{nmel:>12}{mark}")
        print(f" => val {best_thr:.2f}  (val {args.select_on} {best:.4f})")

        print(f"\n test results")
        for name, thr in [("original (0.50)", 0.50), (f"val ({best_thr:.2f})", best_thr)]:
            P = route(p1t, pmt, pnt, thr)
            _, o = macro(yt, P)
            print(f"    {name:<20}" + "  ".join(
                f"{k}={o[k]:.4f}" for k in ["Accuracy", "Sensitivity", "F1", "PPV"]))
            print(f"    {'':<20}MEL prediction {int((P.argmax(1)==1).sum())} "
                  f"( {int((yt==1).sum())})")
        results[tag] = (route(p1t, pmt, pnt, best_thr).argmax(1), yt, idt, best_thr)

    if args.baseline:
        b = load(args.baseline, "test")
        if b is None:
            bf = glob.glob(os.path.join(OUTPUT_ROOT, "..", "runs", "*",
                                        f"{args.baseline}_soft_predictions.npz"))
            print(f"\n[{args.baseline}] test prediction .")
        else:
            p1b, pmb, pnb, yb, idb = b
            pb = route(p1b, pmb, pnb, 0.5).argmax(1)
            print(f"\n{'='*70}\n McNemar vs {args.baseline}\n{'='*70}")
            print(f"{'comparison':<28}{'n10':>6}{'n01':>6}{'exact p':>12}")
            for tag, (pt, yt, idt, thr) in results.items():
                assert np.array_equal(idt, idb), " "
                ca, cb = (pb == yb), (pt == yt)
                n10 = int((ca & ~cb).sum()); n01 = int((~ca & cb).sum())
                p = 1.0 if n10 + n01 == 0 else binomtest(n10, n10 + n01, 0.5).pvalue
                print(f"{tag+' (threshold='+f'{thr:.2f}'+')':<28}{n10:>6}{n01:>6}{p:>12.3g}")
            print("\n  n01 and n10 are the discordant counts of the paired test.")


if __name__ == "__main__":
    main()
