"""Regenerate the figures with a common style (Okabe-Ito palette)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.fig_extra import MARGINS, GROUPS, acc, delta_vs_flat, faith_val, loc_iou
from experiments.fig_lesion import load as lesion_load
from experiments.fig_quant import faith_curves, loc_metrics
from SkinCancer.configs.training_config import OUTPUT_ROOT

ROOT = os.path.dirname(os.path.abspath(OUTPUT_ROOT))
UN, SUP = "#E69F00", "#009E73"
G_DARK, G_MID, G_LIGHT = "#4d4d4d", "#808080", "#bfbfbf"
CLS = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]
LBL = {"orig": "LTV (unsupervised)", "sup0.2_bias_local": "LTV + mask supervision",
       "gc2": "Grad-CAM (stage-2)", "gc4": "Grad-CAM (final)",
       "center": "Centred Gaussian", "random": "Random map"}
CMAP = {"orig": UN, "sup0.2_bias_local": SUP, "gc2": G_DARK, "gc4": G_MID,
        "center": G_MID, "random": G_LIGHT}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.frameon": False, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})

def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} ")

def fig2(out):
    M, C = loc_metrics(), faith_curves()
    order = [n for n in ["sup0.2_bias_local", "center", "gc4", "gc2", "orig"] if n in M]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    marks = [("pointing game", "o", 46), ("energy ratio", "s", 38), ("IoU", "^", 46)]
    y = np.arange(len(order))
    for i, n in enumerate(order):
        vals = M[n]
        ax[0].plot([min(vals), max(vals)], [i, i], color=G_LIGHT, lw=2, zorder=1)
        for j, (lab, mk, sz) in enumerate(marks):
            ax[0].scatter(vals[j], i, marker=mk, s=sz, color=CMAP.get(n, G_MID),
                          edgecolor="white", lw=0.6, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker=mk, ms=np.sqrt(sz), color=G_DARK, label=lab)
               for lab, mk, sz in marks]
    ax[0].legend(handles=handles, fontsize=8, loc="lower right", title="marker = metric",
                 title_fontsize=8)
    ax[0].axvline(M["center"][2], color=G_MID, ls="--", lw=1)
    ax[0].text(M["center"][2] + 0.012, -0.55, "centred-Gaussian IoU",
               fontsize=8, color=G_DARK, va="center")
    ax[0].set_yticks(y); ax[0].set_yticklabels([LBL.get(n, n) for n in order])
    ax[0].set_xlim(0, 1.05); ax[0].set_xlabel("score")
    ax[0].set_title("(a) Overlap with lesion masks", loc="left")
    ax[0].invert_yaxis()

    have = [n for n in ["sup0.2_bias_local", "gc2", "center", "gc4", "random", "orig"]
            if f"{n}_del" in C]
    v = [C[f"{n}_ins"].mean() - C[f"{n}_del"].mean() for n in have]
    o = np.argsort(v)
    yy = np.arange(len(have))
    for k, i in enumerate(o):
        ax[1].plot([0, v[i]], [k, k], color=G_LIGHT, lw=1.6, zorder=1)
        ax[1].scatter(v[i], k, s=64, color=CMAP.get(have[i], G_MID),
                      edgecolor="white", lw=0.8, zorder=3)
        ax[1].text(v[i] + (0.006 if v[i] >= 0 else -0.006), k, f"{v[i]:+.3f}",
                   va="center", ha="left" if v[i] >= 0 else "right", fontsize=8)
    if "random" in have:
        ax[1].axvline(v[have.index("random")], color=G_MID, ls=":", lw=1.2)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_yticks(yy); ax[1].set_yticklabels([LBL.get(have[i], have[i]) for i in o])
    ax[1].set_xlim(min(v) - 0.05, max(v) + 0.05)
    ax[1].set_xlabel("insertion $-$ deletion")
    ax[1].set_title("(b) Faithfulness", loc="left")
    fig.tight_layout()
    save(fig, out, "fig2")

def fig4(out):
    import glob
    fs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "zoom", "zoom2_f*_s42.npz")))
    curves = {}
    if fs:
        parts = [np.load(f, allow_pickle=True) for f in fs]
        y = np.concatenate([p["y_true"] for p in parts]).astype(int)
        g = lambda k: np.concatenate([p[k] for p in parts]).argmax(1)
        b = (g("base") == y).mean()
        for src, lab, c in (("mask", "lesion-mask box", G_DARK), ("ltv", "LTV-attention box", SUP)):
            vv = [((g(f"{src}_m{m}") == y).mean() - b) * 100 if f"{src}_m{m}" in parts[0].files
                  else np.nan for m in MARGINS]
            curves[lab] = (np.array(vv), c)
    costs = []
    zf = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "zoom", "zoom_f*_s42.npz")))
    if zf:
        parts = [np.load(f, allow_pickle=True) for f in zf]
        y = np.concatenate([p["y_true"] for p in parts]).astype(int)
        b = (np.concatenate([p["base"] for p in parts]).argmax(1) == y).mean()
        o_ = (np.concatenate([p["ltvonly"] for p in parts]).argmax(1) == y).mean()
        costs.append(("lesion box only (inference)", (o_ - b) * 100))
    lf = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "localize", "loc_sup0.2_bias_local_f*.npz")))
    if lf:
        parts = [np.load(f, allow_pickle=True) for f in lf]
        lab_ = np.concatenate([p["label"] for p in parts])
        fp = np.concatenate([p["flat_pred"] for p in parts])
        rp = np.concatenate([p["readout_pred"] for p in parts])
        costs.append(("attention bound to lesion", ((rp == lab_).mean() - (fp == lab_).mean()) * 100))
    for nm, lab_ in (("bgdim", "background blurred (training)"),
                     ("lcrop", "lesion-centred crop (training)")):
        r = delta_vs_flat(nm)
        if r:
            costs.append((lab_, r[0]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    if curves:
        for lab, (vv, c) in curves.items():
            ax[0].plot(MARGINS, vv, "o-", color=c, lw=2, ms=5, label=lab)
        ax[0].axhspan(-0.6, 0.6, color=G_LIGHT, alpha=0.4, lw=0)
        ax[0].axhline(0, color="k", lw=0.8)
        lim = max(1.0, np.nanmax([np.nanmax(np.abs(vv)) for vv, _ in curves.values()]) * 1.5)
        ax[0].set_ylim(-lim, lim)
        ax[0].text(MARGINS[-1], 0.66, "re-training noise", ha="right", fontsize=8, color=G_DARK)
        ax[0].set_xlabel("context margin kept around the lesion (%)")
        ax[0].set_ylabel("accuracy change vs full image (pp)")
        ax[0].set_title("(a) How much context to keep", loc="left")
        ax[0].legend(fontsize=8.5, loc="lower right")
    if costs:
        lab_ = [c[0] for c in costs]; val = [c[1] for c in costs]
        o = np.argsort(val)[::-1]
        for k, i in enumerate(o):
            ax[1].plot([0, val[i]], [k, k], color=G_LIGHT, lw=1.6, zorder=1)
            ax[1].scatter(val[i], k, s=66, color=UN, edgecolor="white", lw=0.8, zorder=3)
            ax[1].text(val[i] - 0.12, k, f"{val[i]:.1f} pp", va="center", ha="right", fontsize=8.5)
        ax[1].axvspan(-0.6, 0.6, color=G_LIGHT, alpha=0.4, lw=0)
        ax[1].axvline(0, color="k", lw=0.8)
        ax[1].set_yticks(range(len(val))); ax[1].set_yticklabels([lab_[i] for i in o], fontsize=9)
        ax[1].set_xlim(min(val) - 1.2, 0.9)
        ax[1].set_xlabel("accuracy change (pp)")
        ax[1].set_title("(b) Cost of confining the model to the lesion", loc="left")
    fig.tight_layout()
    save(fig, out, "fig4")

def fig5(out):
    bbs = [("MambaOut-Tiny", OUTPUT_ROOT, "o", "-", -0.045, 1),
           ("ConvNeXt-Tiny", os.path.join(ROOT, "outputs_convnext_tiny"), "s", "--", 0.045, -1)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    for lab, root, mk, ls, dx, sgn in bbs:
        r0, r1 = loc_iou(root, "orig"), loc_iou(root, "sup0.2_bias_local")
        if r0 is None or r1 is None:
            continue
        ax[0].plot([dx, 1 + dx], [r0[0], r1[0]], ls, marker=mk, color=SUP, lw=2, ms=7, label=lab)
        ax[0].annotate(f"{r0[0]:.3f}", (dx, r0[0]), xytext=(-9, 9 * sgn),
                       textcoords="offset points", ha="right", fontsize=8.5)
        ax[0].annotate(f"{r1[0]:.3f}", (1 + dx, r1[0]), xytext=(9, 9 * sgn),
                       textcoords="offset points", ha="left", fontsize=8.5)
        ax[0].axhline(r0[1].get("center", np.nan), color=G_MID, ls=":", lw=1.1)
        f0, f1, fr = (faith_val(root, "orig"), faith_val(root, "sup0.2_bias_local"),
                      faith_val(root, "random"))
        if f0 is not None and f1 is not None:
            ax[1].plot([dx, 1 + dx], [f0, f1], ls, marker=mk, color=SUP, lw=2, ms=7, label=lab)
            ax[1].annotate(f"{f0:+.3f}", (dx, f0), xytext=(-9, 9 * sgn),
                           textcoords="offset points", ha="right", fontsize=8.5)
            ax[1].annotate(f"{f1:+.3f}", (1 + dx, f1), xytext=(11, 16 * sgn),
                           textcoords="offset points", ha="left", fontsize=8.5)
            if fr is not None:
                ax[1].axhline(fr, color=G_MID, ls=":", lw=1.1)
    ax[0].text(1.02, r0[1].get("center", 0.65), "centred Gaussian", fontsize=8, color=G_DARK,
               va="bottom", ha="right")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["unsupervised", "mask-supervised"])
    ax[0].set_xlim(-0.35, 1.35); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("area-matched IoU")
    ax[0].set_title("(a) Overlap with lesion masks", loc="left")
    ax[0].legend(fontsize=8.5, loc="center left")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].text(1.3, 0.004, "random map", fontsize=8, color=G_DARK, ha="right", va="bottom")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["unsupervised", "mask-supervised"])
    ax[1].set_xlim(-0.35, 1.5)
    ax[1].set_ylabel("insertion $-$ deletion")
    ax[1].set_title("(b) Faithfulness", loc="left")
    ax[1].legend(fontsize=8.5, loc="center left")
    fig.tight_layout()
    save(fig, out, "fig5")

def fig6(out):
    if acc("flat") is None:
        print(" fig6: flat missing"); return
    ys, labs, vals, marks = [], [], [], []
    y = 0
    shapes = ["o", "s", "^", "D"]
    for gi, (gname, items) in enumerate(GROUPS):
        got = False
        for m, lab in items:
            r = delta_vs_flat(m) or delta_vs_flat(m, sub="explore")
            if r is None:
                continue
            d, nf = r
            ys.append(y); labs.append(lab if nf == 5 else f"{lab}  ({nf}/5 folds)")
            vals.append(d); marks.append((shapes[gi], gname)); y += 1; got = True
        if got:
            ys.append(y); labs.append(""); vals.append(np.nan); marks.append((None, None)); y += 1
    fig, ax = plt.subplots(figsize=(7.8, 0.33 * len(vals) + 1.5))
    ax.axvspan(-0.6, 0.6, color=G_LIGHT, alpha=0.45, lw=0)
    ax.axvline(0, color="k", lw=0.9)
    seen = set()
    for v, yy, (mk, gname) in zip(vals, ys, marks):
        if mk is None or np.isnan(v):
            continue
        ax.plot([0, v], [yy, yy], color=G_LIGHT, lw=1.1, zorder=1)
        ax.scatter(v, yy, marker=mk, s=52, color=UN if v < -0.6 else G_DARK,
                   edgecolor="white", lw=0.7, zorder=3,
                   label=gname if gname not in seen else None)
        seen.add(gname)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel("accuracy change vs single-stage classifier (pp)")
    ax.set_title("Every design variant tested, against the re-training noise band", loc="left")
    ax.text(0.65, max(ys), "re-training noise", fontsize=8, color=G_DARK, va="bottom")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4)
    fig.tight_layout()
    save(fig, out, "fig6")

def fig7(out):
    un, sp = lesion_load("orig"), lesion_load("sup0.2_bias_local")
    if un is None or sp is None:
        print(" fig7: results "); return
    gi = lambda d, w: d["R"][:, d["names"].index(w), 2]
    n = min(len(un["area"]), len(sp["area"]))
    v = {"un": gi(un, "ltv")[:n], "sup": gi(sp, "ltv")[:n],
         "cen": gi(un, "center")[:n], "gc": gi(un, "gc2")[:n]}
    area, lab, pred = un["area"][:n], un["label"][:n], un["pred"][:n]

    fig = plt.figure(figsize=(12.4, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
    a0, a1, a2, a3 = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                      fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))

    q = np.quantile(area, [1/3, 2/3])
    bins = [area < q[0], (area >= q[0]) & (area < q[1]), area >= q[1]]
    xs = np.arange(3)
    for k, lab_, c, ls in (("sup", "LTV + mask supervision", SUP, "-"),
                           ("cen", "Centred Gaussian", G_MID, ":"),
                           ("gc", "Grad-CAM (stage-2)", G_DARK, "-."),
                           ("un", "LTV (unsupervised)", UN, "-")):
        a0.plot(xs, [np.nanmean(v[k][m]) for m in bins], ls, marker="o", color=c, lw=2, ms=6,
                label=lab_)
    a0.set_xticks(xs)
    a0.set_xticklabels([f"small\n(<{q[0]:.2f})", "medium", f"large\n(>{q[1]:.2f})"], fontsize=9)
    a0.set_ylim(0, 1); a0.set_ylabel("area-matched IoU")
    a0.set_title("(a) By lesion size", loc="left")
    a0.legend(fontsize=8, loc="center right")

    ys = np.arange(7)
    for c in range(7):
        m = lab == c
        u_, s_, ce = np.nanmean(v["un"][m]), np.nanmean(v["sup"][m]), np.nanmean(v["cen"][m])
        a1.plot([u_, s_], [c, c], color=G_LIGHT, lw=2.2, zorder=1)
        a1.scatter(u_, c, s=52, color=UN, edgecolor="white", lw=0.6, zorder=3,
                   label="LTV (unsupervised)" if c == 0 else None)
        a1.scatter(s_, c, s=52, color=SUP, edgecolor="white", lw=0.6, zorder=3,
                   label="LTV + mask supervision" if c == 0 else None)
        a1.scatter(ce, c, marker="|", s=90, color=G_MID, zorder=2,
                   label="Centred Gaussian" if c == 0 else None)
    a1.set_yticks(ys); a1.set_yticklabels(CLS, fontsize=9); a1.invert_yaxis()
    a1.set_xlim(0, 1); a1.set_xlabel("area-matched IoU")
    a1.set_title("(b) By diagnostic class", loc="left")
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    ok = pred == lab
    for i, (k, lab_, c) in enumerate((("un", "LTV (unsupervised)", UN),
                                      ("cen", "Centred Gaussian", G_MID),
                                      ("sup", "LTV + mask supervision", SUP))):
        m1, m2 = np.nanmean(v[k][ok]), np.nanmean(v[k][~ok])
        a2.plot([0, 1], [m1, m2], "-", marker="o", color=c, lw=2, ms=6, label=lab_)
        a2.annotate(f"{m1:.3f}", (0, m1), xytext=(-8, 0), textcoords="offset points",
                    ha="right", fontsize=8)
        a2.annotate(f"{m2:.3f}", (1, m2), xytext=(8, 0), textcoords="offset points",
                    ha="left", fontsize=8)
    a2.set_xticks([0, 1])
    a2.set_xticklabels([f"correct (n={int(ok.sum())})", f"misclassified (n={int((~ok).sum())})"],
                       fontsize=9)
    a2.set_xlim(-0.4, 1.4); a2.set_ylim(0, 1); a2.set_ylabel("area-matched IoU")
    a2.set_title("(c) Correct vs misclassified", loc="left")
    a2.legend(fontsize=8, loc="center left")

    idx = np.random.default_rng(0).choice(n, size=min(n, 2200), replace=False)
    a3.scatter(area[idx], v["un"][idx], s=5, alpha=0.20, color=UN, lw=0)
    a3.scatter(area[idx], v["sup"][idx], s=5, alpha=0.20, color=SUP, lw=0)
    xs = np.linspace(area.min(), area.max(), 14)
    for k, c, ls, lab_ in (("sup", SUP, "-", "LTV + mask supervision"),
                           ("cen", G_MID, ":", "Centred Gaussian"),
                           ("un", UN, "-", "LTV (unsupervised)")):
        mids, mean = [], []
        for lo, hi in zip(xs[:-1], xs[1:]):
            m = (area >= lo) & (area < hi)
            if m.sum() > 20:
                mids.append((lo + hi) / 2); mean.append(np.nanmean(v[k][m]))
        a3.plot(mids, mean, ls, color=c, lw=2.4, label=lab_)
    a3.set_xlabel("lesion area (fraction of image)"); a3.set_ylabel("area-matched IoU")
    a3.set_ylim(0, 1)
    a3.set_title("(d) IoU versus lesion size", loc="left")
    a3.legend(fontsize=8, loc="upper left")
    save(fig, out, "fig7")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["2", "4", "5", "6", "7"])
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for w in a.which:
        {"2": fig2, "4": fig4, "5": fig5, "6": fig6, "7": fig7}[w](a.out)

if __name__ == "__main__":
    main()
