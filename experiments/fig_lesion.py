"""Figure: explanation quality by lesion size, class, and prediction correctness."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from SkinCancer.configs.training_config import OUTPUT_ROOT

LOC = os.path.join(OUTPUT_ROOT, "localize")
CLS = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]
COL = {"unsup": "#c0392b", "sup": "#1f6f3f", "center": "#b8860b", "gc": "#34495e"}


def load(name, seeds=(42, 43, 44)):
    """Collect {metrics, label, flat_pred, lesion_area, names} over seeds and folds."""
    out = []
    for sd in seeds:
        base = LOC if sd == 42 else os.path.join(LOC, f"seed{sd}")
        fs = [os.path.join(base, f"loc_{name}_f{k}.npz") for k in range(5)]
        fs = [f for f in fs if os.path.exists(f)]
        if len(fs) < 5:
            continue
        out += [np.load(f, allow_pickle=True) for f in fs]
    if not out:
        return None
    names = list(out[0]["names"])
    return dict(R=np.concatenate([p["metrics"] for p in out]),
                label=np.concatenate([p["label"] for p in out]),
                pred=np.concatenate([p["flat_pred"] for p in out]),
                area=np.concatenate([p["lesion_area"] for p in out]),
                names=names)


def iou(d, which):
    j = d["names"].index(which) if which != "ltv" else d["names"].index("ltv")
    return d["R"][:, j, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/fig7")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    un, sp = load("orig"), load("sup0.2_bias_local")
    if un is None or sp is None:
        print("results (orig / sup0.2_bias_local 5fold )"); return

    v = {"unsup": iou(un, "ltv"), "sup": iou(sp, "ltv"),
         "center": iou(un, "center"), "gc": iou(un, "gc2")}
    area, lab, pred = un["area"], un["label"], un["pred"]
    n = min(len(x) for x in v.values())
    v = {k: x[:n] for k, x in v.items()}
    area, lab, pred = area[:n], lab[:n], pred[:n]
    LBL = {"unsup": "LTV (unsupervised)", "sup": "LTV + mask supervision",
           "center": "Centred Gaussian", "gc": "Grad-CAM"}

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))

    q = np.quantile(area, [1/3, 2/3])
    bins = [(area < q[0], f"small\n(<{q[0]:.2f})"), ((area >= q[0]) & (area < q[1]), f"medium"),
            (area >= q[1], f"large\n(>{q[1]:.2f})")]
    x = np.arange(3); w = 0.2
    for i, k in enumerate(["unsup", "gc", "center", "sup"]):
        ax[0, 0].bar(x + (i - 1.5) * w, [np.nanmean(v[k][m]) for m, _ in bins], w,
                     label=LBL[k], color=COL[k], edgecolor="white")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels([t for _, t in bins], fontsize=9)
    ax[0, 0].set_ylabel("area-matched IoU"); ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("(a) By lesion size", fontsize=11, loc="left")
    ax[0, 0].legend(fontsize=8, frameon=False, ncol=2)

    xs = np.arange(7)
    for i, k in enumerate(["unsup", "center", "sup"]):
        ax[0, 1].bar(xs + (i - 1) * 0.26, [np.nanmean(v[k][lab == c]) for c in range(7)], 0.26,
                     label=LBL[k], color=COL[k], edgecolor="white")
    ax[0, 1].set_xticks(xs); ax[0, 1].set_xticklabels(CLS, fontsize=9)
    ax[0, 1].set_ylabel("area-matched IoU"); ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_title("(b) By diagnostic class", fontsize=11, loc="left")
    ax[0, 1].legend(fontsize=8, frameon=False)

    ok = pred == lab
    xs = np.arange(2); txt = []
    for i, k in enumerate(["unsup", "center", "sup"]):
        m = [np.nanmean(v[k][ok]), np.nanmean(v[k][~ok])]
        ax[1, 0].bar(xs + (i - 1) * 0.26, m, 0.26, label=LBL[k], color=COL[k], edgecolor="white")
        try:
            p = mannwhitneyu(v[k][ok], v[k][~ok]).pvalue
        except ValueError:
            p = np.nan
        txt.append(f"{LBL[k]}: {m[0]:.3f} vs {m[1]:.3f} (p={p:.1g})")
    ax[1, 0].set_xticks(xs)
    ax[1, 0].set_xticklabels([f"correct\n(n={ok.sum()})", f"misclassified\n(n={(~ok).sum()})"], fontsize=9)
    ax[1, 0].set_ylabel("area-matched IoU"); ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_title("(c) Correct vs misclassified predictions", fontsize=11, loc="left")
    ax[1, 0].legend(fontsize=8, frameon=False)

    idx = np.random.default_rng(0).choice(n, size=min(n, 2500), replace=False)
    for k in ("unsup", "sup"):
        ax[1, 1].scatter(area[idx], v[k][idx], s=5, alpha=0.25, color=COL[k], label=LBL[k], lw=0)
    xs = np.linspace(area.min(), area.max(), 12)
    for k in ("unsup", "sup", "center"):
        mids, means = [], []
        for lo, hi in zip(xs[:-1], xs[1:]):
            m = (area >= lo) & (area < hi)
            if m.sum() > 20:
                mids.append((lo + hi) / 2); means.append(np.nanmean(v[k][m]))
        ax[1, 1].plot(mids, means, lw=2.2, color=COL[k], label=LBL[k] + " (mean)" if k == "center" else None)
    ax[1, 1].set_xlabel("lesion area (fraction of image)"); ax[1, 1].set_ylabel("area-matched IoU")
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_title("(d) IoU versus lesion size", fontsize=11, loc="left")
    ax[1, 1].legend(fontsize=8, frameon=False, markerscale=2)

    for row in ax:
        for a_ in row:
            a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight")
    print(f": {a.out}.pdf / .png   n={n}")
    print(" (a) IoU by lesion size")
    for k in ("unsup", "center", "sup"):
        print(f"      {LBL[k]:<24}" + "  ".join(f"{np.nanmean(v[k][m]):.3f}" for m, _ in bins))
    print(" (c) correct versus misclassified")
    for t in txt:
        print("      " + t)


if __name__ == "__main__":
    main()
