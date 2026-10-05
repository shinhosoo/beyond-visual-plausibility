"""Additional statistics (no GPU required; reads saved predictions).

  1) Equivalence: confidence interval and TOST for the correction configuration
  2) Sample unit: explanation comparisons summarised per fold and seed
  3) Multiple comparisons: Holm-corrected p-values for the design variants

  python -m experiments.stats_extra
"""
import glob
import os

import numpy as np
from scipy import stats

from SkinCancer.configs.training_config import OUTPUT_ROOT

ROOT = os.path.dirname(os.path.abspath(OUTPUT_ROOT))
F7 = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
SEEDS = [42, 43, 44, 45, 46]

def ld(p):
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None

def equivalence(margin=0.02):
    """Standard split, five seeds: 95% CI of the difference and TOST against the margin."""
    print("=" * 78)
    print(f" 1) Equivalence testing (margin +/-{margin*100:.0f} pp, standard split, five seeds)")
    print("=" * 78)
    base = []
    for s in SEEDS:
        d = ld(os.path.join(OUTPUT_ROOT, "hier",
                            ("flat" if s == 42 else f"flat_s{s}") + "_test_predictions.npz"))
        if d is not None:
            base.append(((d["y_prob"].argmax(1) == d["y_true"]).mean(), s))
    for model, label in (("d3", "correction (base lr)"), ("d3lr", "correction (head lr x10)")):
        rows = []
        for a, s in base:
            d = ld(os.path.join(OUTPUT_ROOT, "residual", f"{model}_full_s{s}_test_predictions.npz"))
            if d is not None:
                rows.append(((d["y_prob"].argmax(1) == d["y_true"]).mean(), a))
        if len(rows) < 3:
            print(f"  {label:<26} seed ({len(rows)})"); continue
        d_ = np.array([m - b for m, b in rows])
        n = len(d_); mean = d_.mean(); se = d_.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, n - 1)
        lo, hi = mean - tcrit * se, mean + tcrit * se
        t1 = (mean + margin) / se; t2 = (mean - margin) / se
        p_tost = max(1 - stats.t.cdf(t1, n - 1), stats.t.cdf(t2, n - 1))
        verdict = " (TOST)" if p_tost < 0.05 else " "
        print(f"  {label:<26} {mean*100:+.2f}%p  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]%p"
              f"  TOST p={p_tost:.3f}  -> {verdict}")
    print("  If the interval lies inside the margin, the two are practically equivalent.")

def fold_level():
    print("\n" + "=" * 78)
    print(" 2) Explanation comparisons summarised per fold and seed")
    print("=" * 78)
    rows = {}
    for name in ("orig", "sup0.2_bias_local"):
        vals = []
        for sd in (42, 43, 44):
            base = os.path.join(OUTPUT_ROOT, "localize") if sd == 42 else \
                os.path.join(OUTPUT_ROOT, "localize", f"seed{sd}")
            for k in range(5):
                f = os.path.join(base, f"loc_{name}_f{k}.npz")
                d = ld(f)
                if d is None:
                    continue
                nm = list(d["names"]); L = nm.index("ltv")
                vals.append((np.nanmean(d["metrics"][:, L, 2]),
                             np.nanmean(d["metrics"][:, nm.index("center"), 2]),
                             np.nanmean(d["metrics"][:, nm.index("gc2"), 2]), sd, k))
        rows[name] = vals
    if not rows.get("orig") or not rows.get("sup0.2_bias_local"):
        print(" results "); return
    for name, label in (("orig", "LTV (unsupervised)"), ("sup0.2_bias_local", "LTV + mask sup.")):
        v = np.array([[a, b, c] for a, b, c, _, _ in rows[name]])
        n = len(v)
        for j, ref in ((1, "centred Gaussian"), (2, "Grad-CAM (stage-2)")):
            d_ = v[:, 0] - v[:, j]
            t, p = stats.ttest_rel(v[:, 0], v[:, j])
            w = stats.wilcoxon(v[:, 0], v[:, j]).pvalue if n >= 6 else np.nan
            print(f"  {label:<20} vs {ref:<20} Δ{d_.mean():+.3f} "
                  f"(n={n} folds)  paired t p={p:.2g}  Wilcoxon p={w:.2g}")
    print(" prediction p evaluation.")
    print("  Comparisons summarised per fold and seed.")

def multiple_comparisons():
    print("\n" + "=" * 78)
    print(" 3) cross-validation comparison Holm correction")
    print("=" * 78)
    from experiments.fig_extra import delta_vs_flat
    from scipy.stats import binomtest
    models = ["d1", "d1lr", "d1p", "d3lr", "midA", "midB", "lcrop", "bgdim", "h1sup",
              "ltv", "ltvb", "ltvn", "gap_la_ema", "k2_bias_local"]
    res = []
    for m in models:
        for sub in ("residual", "explore"):
            pm, ym, pf = [], [], []
            for k in range(5):
                f1 = os.path.join(OUTPUT_ROOT, sub, f"{m}_f{k}_s42_val_predictions.npz")
                f2 = os.path.join(OUTPUT_ROOT, "residual", f"flat_f{k}_s42_val_predictions.npz")
                if not (os.path.exists(f1) and os.path.exists(f2)):
                    continue
                d1, d2 = np.load(f1, allow_pickle=True), np.load(f2, allow_pickle=True)
                o1, o2 = np.argsort(d1["image_id"]), np.argsort(d2["image_id"])
                pm.append(d1["y_prob"][o1].argmax(1)); ym.append(d1["y_true"][o1].astype(int))
                pf.append(d2["y_prob"][o2].argmax(1))
            if not pm:
                continue
            y, a, b = np.concatenate(ym), np.concatenate(pf), np.concatenate(pm)
            n10 = int(((a == y) & (b != y)).sum()); n01 = int(((a != y) & (b == y)).sum())
            p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
            r = delta_vs_flat(m, sub=sub)
            res.append((m, r[0] if r else np.nan, p, n01 > n10))
            break
    if not res:
        print(" results missing"); return
    ps = np.array([r[2] for r in res])
    order = np.argsort(ps)
    holm = np.empty_like(ps)
    m_ = len(ps)
    prev = 0
    for rank, i in enumerate(order):
        val = min(1.0, (m_ - rank) * ps[i])
        prev = max(prev, val)
        holm[i] = prev
    print(f"  {'variant':<16}{'Δacc (pp)':>11}{'raw p':>11}{'Holm p':>11}   direction")
    for (m, d, p, better), hp in zip(res, holm):
        print(f"  {m:<16}{d:>11.2f}{p:>11.2g}{hp:>11.2g}   "
              f"{'variant better' if better else 'flat better'}")
    print(f" correction : "
          f"{sum(1 for (m, d, p, b), hp in zip(res, holm) if b and hp < 0.05)}")

if __name__ == "__main__":
    equivalence()
    fold_level()
    multiple_comparisons()
