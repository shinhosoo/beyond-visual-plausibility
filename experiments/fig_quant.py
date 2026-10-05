"""Quantitative figure: overlap with the lesion masks and faithfulness."""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from SkinCancer.configs.training_config import OUTPUT_ROOT

LOC = os.path.join(OUTPUT_ROOT, "localize")
FAI = os.path.join(OUTPUT_ROOT, "faithful")
LAB = {"orig": "LTV (unsupervised)", "sup0.2_bias_local": "LTV + mask supervision",
       "gc4": "Grad-CAM (final)", "gc2": "Grad-CAM (stage-2)",
       "center": "Centred Gaussian", "random": "Random"}
COL = {"orig": "#c0392b", "sup0.2_bias_local": "#1f6f3f", "gc4": "#7f8c8d",
       "gc2": "#34495e", "center": "#b8860b", "random": "#95a5a6"}

def loc_metrics():
    """Return {name: (pointing, energy, IoU)} for the LTV map and the reference maps."""
    out, ref = {}, None
    for name in ("orig", "sup0.2_bias_local"):
        fs = [os.path.join(LOC, f"loc_{name}_f{k}.npz") for k in range(5)]
        for sd in (43, 44):
            fs += [os.path.join(LOC, f"seed{sd}", f"loc_{name}_f{k}.npz") for k in range(5)]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            continue
        parts = [np.load(f, allow_pickle=True) for f in fs]
        R = np.concatenate([p["metrics"] for p in parts])
        names = list(parts[0]["names"])
        L = names.index("ltv")
        out[name] = tuple(np.nanmean(R[:, L, i]) for i in range(3))
        if ref is None:
            ref = {n: tuple(np.nanmean(R[:, j, i]) for i in range(3))
                   for j, n in enumerate(names) if n != "ltv"}
    out.update(ref or {})
    return out

def faith_curves():
    fs = sorted(glob.glob(os.path.join(FAI, "faith_f*_s*.npz")))
    if not fs:
        return {}
    parts = [np.load(f, allow_pickle=True) for f in fs]
    keys = [k for k in parts[0].files if k.endswith("_del") or k.endswith("_ins")]
    return {k: np.concatenate([p[k] for p in parts if k in p.files]).mean(0) for k in keys}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/fig2")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    M, C = loc_metrics(), faith_curves()
    print(" overlap results:", ", ".join(sorted(M)))
    print(" faithfulness results:", ", ".join(sorted({k.rsplit("_", 1)[0] for k in C})) or "missing")
    if not M:
        print(" measurement results missing"); return

    have_f = {k.rsplit("_", 1)[0] for k in C}
    order = [n for n in ["orig", "sup0.2_bias_local", "gc2", "gc4", "center", "random"]
             if n in M or n in have_f]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))

    w, xs = 0.26, np.arange(len(order))
    order_a = [n for n in order if n in M]
    xs = np.arange(len(order_a))
    for i, (mi, lab) in enumerate([(0, "Pointing game"), (1, "Energy ratio"), (2, "Area-matched IoU")]):
        ax[0].bar(xs + (i - 1) * w, [M[n][mi] for n in order_a], w, label=lab,
                  color=["#4c72b0", "#dd8452", "#55a868"][i], edgecolor="white", linewidth=0.6)
    ax[0].axhline(M["center"][2], ls="--", c="#b8860b", lw=1.2)
    ax[0].text(len(order_a) - 0.45, M["center"][2] + 0.015, "centred-Gaussian IoU",
               ha="right", fontsize=8, color="#b8860b")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([LAB.get(n, n).replace(" (", "\n(") for n in order_a], fontsize=8)
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("score")
    ax[0].set_title("(a) Overlap with lesion masks", fontsize=11, loc="left")
    ax[0].legend(fontsize=8, frameon=False, ncol=3)

    have = [n for n in order if f"{n}_del" in C]
    if have:
        vals = [C[f"{n}_ins"].mean() - C[f"{n}_del"].mean() for n in have]
        o2 = np.argsort(vals)
        ax[1].barh(np.arange(len(have)), [vals[i] for i in o2],
                   color=[COL.get(have[i], "#888") for i in o2], edgecolor="white", height=0.62)
        ax[1].set_yticks(np.arange(len(have)))
        ax[1].set_yticklabels([LAB.get(have[i], have[i]) for i in o2], fontsize=9)
        ax[1].axvline(0, color="k", lw=0.8)
        rnd = vals[have.index("random")] if "random" in have else None
        if rnd is not None:
            ax[1].axvline(rnd, ls="--", c="#95a5a6", lw=1.2)
            ax[1].text(rnd, len(have) - 0.35, " random map", fontsize=8, color="#7f8c8d", va="top")
        for k, i in enumerate(o2):
            ax[1].text(vals[i] + (0.004 if vals[i] >= 0 else -0.004), k, f"{vals[i]:+.3f}",
                       va="center", ha="left" if vals[i] >= 0 else "right", fontsize=8)
        ax[1].set_xlabel("insertion $-$ deletion  (higher = more faithful)")
        ax[1].set_title("(b) Faithfulness", fontsize=11, loc="left")
        ax[1].set_xlim(min(min(vals) - 0.03, -0.04), max(vals) + 0.05)
    else:
        ax[1].text(0.5, 0.5, "faithfulness results not found", ha="center")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight")

    if C:
        f2, b = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
        x = np.linspace(0, 100, len(next(iter(C.values()))))
        for n in order:
            if f"{n}_del" not in C:
                continue
            b[0].plot(x, C[f"{n}_del"], color=COL.get(n, "k"), lw=1.8, label=LAB.get(n, n))
            b[1].plot(x, C[f"{n}_ins"], color=COL.get(n, "k"), lw=1.8)
        b[0].set_title("deletion (lower is better)", fontsize=10, loc="left")
        b[1].set_title("insertion (higher is better)", fontsize=10, loc="left")
        for a_ in b:
            a_.set_xlabel("% of pixels"); a_.spines[["top", "right"]].set_visible(False)
        b[0].set_ylabel("probability of predicted class")
        b[0].legend(fontsize=8, frameon=False)
        f2.tight_layout()
        for ext in ("pdf", "png"):
            f2.savefig(f"{a.out}_curves.{ext}", dpi=220, bbox_inches="tight")
        print(f": {a.out}_curves.pdf / .png ()")
    print(f": {a.out}.pdf / .png")
    print("  " + "  ".join(f"{n}: IoU {M[n][2]:.3f}" for n in order if n in M))

if __name__ == "__main__":
    main()
