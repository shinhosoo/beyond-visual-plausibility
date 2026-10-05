"""Class-wise metrics, Stage-1 calibration and MEL-NV confusion (no GPU required).

  1) Class-wise performance of the single-stage classifier over seeds
  2) Calibration of the Stage-1 routing probabilities, with a reliability diagram
  3) MEL-NV confusion and paired McNemar tests with Holm correction

  python -m experiments.paper_tables --out figs
"""
import argparse
import os

import numpy as np
from scipy.stats import binomtest

from SkinCancer.configs.training_config import OUTPUT_ROOT

CLS = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]
SEEDS = [42, 43, 44, 45, 46]
R = OUTPUT_ROOT

def ld(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None

def route(d):
    p1, pm, pn = d["p_stage1"], d["p_mel"], d["p_nonmel"]
    o = np.zeros((len(p1), 7))
    o[:, 0] = p1 * (1 - pm); o[:, 1] = p1 * pm; o[:, 2:] = (1 - p1)[:, None] * pn
    return o

def preds(name, s):
    """name: flat | hard | soft | teme | ltv | full"""
    if name == "flat":
        d = ld(os.path.join(R, "hier", ("flat" if s == 42 else f"flat_s{s}") + "_test_predictions.npz"))
        return (d["y_prob"], d["y_true"].astype(int)) if d is not None else None
    runs = os.path.join(os.path.dirname(os.path.abspath(R)), "runs")
    if s == 42:
        key = {"soft": "abl_baseline_soft", "hard": "abl_baseline_hard",
               "teme": "abl_teme_soft", "ltv": "abl_ltv_soft", "full": "abl_full_soft"}.get(name)
        if key:
            f = os.path.join(runs, key.rsplit("_", 1)[0], f"{key}_predictions.npz")
            d = ld(f)
            if d is not None:
                return d["y_prob"], d["y_true"].astype(int)
    tag = {"soft": "baseline", "hard": "baseline", "teme": "teme", "ltv": "ltv", "full": "full"}[name]
    for c in ([os.path.join(R, "experiments", f"seed{s}_full_test_predictions.npz")] if name == "full" else []) + \
             [os.path.join(R, "experiments", f"tbl_{tag}_s{s}_test_predictions.npz")]:
        d = ld(c)
        if d is not None:
            return route(d), d["y_true"].astype(int)
    return None

def class_wise():
    print("=" * 96)
    print(" 1) Class-wise performance of the single-stage classifier (mean +/- s.d., five seeds)")
    print("=" * 96)
    acc = {c: [] for c in range(7)}
    rec, pre, f1s, sup = dict(acc), {c: [] for c in range(7)}, {c: [] for c in range(7)}, {}
    rec = {c: [] for c in range(7)}
    for s in SEEDS:
        r = preds("flat", s)
        if r is None:
            continue
        P, y = r
        p = P.argmax(1)
        for c in range(7):
            tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum()
            fn = ((p != c) & (y == c)).sum()
            rec[c].append(tp / max(tp + fn, 1)); pre[c].append(tp / max(tp + fp, 1))
            f1s[c].append(2 * tp / max(2 * tp + fp + fn, 1)); sup[c] = int((y == c).sum())
    if not rec[0]:
        print(" prediction missing"); return
    print(f"  {'class':<8}{'support':>9}{'recall':>18}{'precision':>18}{'F1':>18}")
    for c in range(7):
        print(f"  {CLS[c]:<8}{sup[c]:>9}"
              f"{np.mean(rec[c]):>12.3f}±{np.std(rec[c], ddof=1):.3f}"
              f"{np.mean(pre[c]):>12.3f}±{np.std(pre[c], ddof=1):.3f}"
              f"{np.mean(f1s[c]):>12.3f}±{np.std(f1s[c], ddof=1):.3f}")
    print(f"  {'macro':<8}{sum(sup.values()):>9}"
          f"{np.mean([np.mean(rec[c]) for c in range(7)]):>12.3f}       "
          f"{np.mean([np.mean(pre[c]) for c in range(7)]):>12.3f}       "
          f"{np.mean([np.mean(f1s[c]) for c in range(7)]):>12.3f}")

def calibration(out):
    print("\n" + "=" * 96)
    print(" 2) Calibration of the Stage-1 routing probabilities")
    print("=" * 96)
    rows = []
    for s in SEEDS:
        d = ld(os.path.join(R, "experiments", f"tbl_baseline_s{s}_test_predictions.npz"))
        if d is None:
            continue
        p1 = np.asarray(d["p_stage1"], dtype=float)
        y = d["y_true"].astype(int)
        mel = np.isin(y, [0, 1]).astype(int)
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p1, bins) - 1, 0, 9)
        ece, curve = 0.0, []
        for b in range(10):
            m = idx == b
            if m.sum() == 0:
                curve.append((np.nan, np.nan, 0)); continue
            conf, acc_ = p1[m].mean(), mel[m].mean()
            ece += m.mean() * abs(acc_ - conf)
            curve.append((conf, acc_, int(m.sum())))
        brier = np.mean((p1 - mel) ** 2)
        rows.append((s, ece, brier, curve))
        print(f"  seed {s}:  ECE {ece:.4f}   Brier {brier:.4f}   n={len(y)}")
    if not rows:
        print(" Stage-1 missing"); return
    e = [r[1] for r in rows]; b = [r[2] for r in rows]
    print(f" : ECE {np.mean(e):.4f} ± {np.std(e, ddof=1):.4f}"
          f"   Brier {np.mean(b):.4f} ± {np.std(b, ddof=1):.4f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
        for s, _, _, curve in rows:
            xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
            ax.plot(xs, ys, "o-", ms=4, lw=1.4, alpha=0.75, label=f"seed {s}")
        ax.set_xlabel("mean predicted probability (melanocytic)")
        ax.set_ylabel("observed fraction")
        ax.set_title("Stage-1 routing calibration", loc="left", fontsize=11)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"fig_calib.{ext}"), dpi=220, bbox_inches="tight")
        print(f" : {out}/fig_calib.png")
    except Exception as ex:
        print(f" figure : {ex}")

def mel_nv():
    print("\n" + "=" * 96)
    print(" 3) MEL-NV (5seed) McNemar (seed 42, Holm correction)")
    print("=" * 96)
    names = [("flat", "1-stage"), ("hard", "2-stage hard"), ("soft", "2-stage soft"),
             ("full", "+TEME+LTV")]
    stat = {}
    print(f"  {'model':<16}{'MEL recall':>20}{'MEL->NV':>20}{'NV->MEL':>20}")
    for key, lab in names:
        rr, mn, nm = [], [], []
        for s in SEEDS:
            r = preds(key, s)
            if r is None:
                continue
            P, y = r; p = P.argmax(1)
            rr.append((p[y == 1] == 1).mean())
            mn.append((p[y == 1] == 0).mean())
            nm.append((p[y == 0] == 1).mean())
        if not rr:
            print(f"  {lab:<16} prediction missing"); continue
        stat[key] = (rr, mn, nm)
        f = lambda v: f"{100*np.mean(v):>10.1f} ± {100*np.std(v, ddof=1):.1f}%"
        print(f"  {lab:<16}{f(rr):>20}{f(mn):>20}{f(nm):>20}")

    pairs = [("flat", "soft"), ("flat", "full"), ("soft", "full")]
    out = []
    for a, b in pairs:
        ra, rb = preds(a, 42), preds(b, 42)
        if ra is None or rb is None:
            continue
        (Pa, y), (Pb, _) = ra, rb
        pa, pb = Pa.argmax(1), Pb.argmax(1)
        for out_lab, mask, cond in (("MEL recall", y == 1, lambda p: p == 1),
                                    ("MEL->NV", y == 1, lambda p: p == 0),
                                    ("NV->MEL", y == 0, lambda p: p == 1)):
            oa, ob = cond(pa[mask]), cond(pb[mask])
            n10 = int((oa & ~ob).sum()); n01 = int((~oa & ob).sum())
            pv = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
            out.append((out_lab, f"{a} vs {b}", n10, n01, pv))
    print(f"\n  {'outcome':<12}{'comparison':<18}{'n10':>6}{'n01':>6}{'raw p':>10}{'Holm p':>10}")
    for fam in ("MEL recall", "MEL->NV", "NV->MEL"):
        rows = [r for r in out if r[0] == fam]
        ps = np.array([r[4] for r in rows]); m = len(ps)
        order = np.argsort(ps); prev = 0; holm = np.empty(m)
        for rank, i in enumerate(order):
            prev = max(prev, min(1.0, (m - rank) * ps[i])); holm[i] = prev
        for (lab, cmp_, n10, n01, p), hp in zip(rows, holm):
            print(f"  {lab:<12}{cmp_:<18}{n10:>6}{n01:>6}{p:>10.3g}{hp:>10.3g}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    class_wise()
    calibration(a.out)
    mel_nv()

if __name__ == "__main__":
    main()
