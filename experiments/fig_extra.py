"""Figures: the accuracy cost of lesion alignment, backbone replication, design variants."""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from SkinCancer.configs.training_config import OUTPUT_ROOT

ROOT = os.path.dirname(os.path.abspath(OUTPUT_ROOT))
MARGINS = [0, 20, 40, 60, 80, 120]


def oof(model, sub="residual", root=None, seed=42):
    root = root or OUTPUT_ROOT
    P, Y = [], []
    for k in range(5):
        f = os.path.join(root, sub, f"{model}_f{k}_s{seed}_val_predictions.npz")
        if not os.path.exists(f):
            return None
        d = np.load(f, allow_pickle=True)
        P.append(d["y_prob"].argmax(1)); Y.append(d["y_true"])
    return np.concatenate(P), np.concatenate(Y).astype(int)


def acc(model, **kw):
    r = oof(model, **kw)
    return None if r is None else float((r[0] == r[1]).mean())


def delta_vs_flat(model, sub="residual", seed=42, min_folds=2):
    """ fold flat comparison. ( pp, fold )"""
    ks, pm, ym, pf = [], [], [], []
    for k in range(5):
        f1 = os.path.join(OUTPUT_ROOT, sub, f"{model}_f{k}_s{seed}_val_predictions.npz")
        f2 = os.path.join(OUTPUT_ROOT, "residual", f"flat_f{k}_s{seed}_val_predictions.npz")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        d1 = np.load(f1, allow_pickle=True); d2 = np.load(f2, allow_pickle=True)
        o1 = np.argsort(d1["image_id"]); o2 = np.argsort(d2["image_id"])
        pm.append(d1["y_prob"][o1].argmax(1)); ym.append(d1["y_true"][o1].astype(int))
        pf.append(d2["y_prob"][o2].argmax(1)); ks.append(k)
    if len(ks) < min_folds:
        return None
    y = np.concatenate(ym)
    return ((np.concatenate(pm) == y).mean() - (np.concatenate(pf) == y).mean()) * 100, len(ks)


def fig4(out):
    base = acc("flat")
    if base is None:
        print(" [fig4] flat results missing"); return
    fs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "zoom", "zoom2_f*_s42.npz")))
    curves = {}
    if fs:
        parts = [np.load(f, allow_pickle=True) for f in fs]
        y = np.concatenate([p["y_true"] for p in parts]).astype(int)
        g = lambda k: np.concatenate([p[k] for p in parts]).argmax(1)
        b = (g("base") == y).mean()
        for src, lab in (("mask", "lesion-mask box"), ("ltv", "LTV-attention box")):
            v = []
            for m in MARGINS:
                k = f"{src}_m{m}"
                v.append((g(k) == y).mean() - b if k in parts[0].files else np.nan)
            curves[lab] = np.array(v) * 100
    costs = []
    zf = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "zoom", "zoom_f*_s42.npz")))
    if zf:
        parts = [np.load(f, allow_pickle=True) for f in zf]
        y = np.concatenate([p["y_true"] for p in parts]).astype(int)
        b = (np.concatenate([p["base"] for p in parts]).argmax(1) == y).mean()
        o = (np.concatenate([p["ltvonly"] for p in parts]).argmax(1) == y).mean()
        costs.append(("lesion box only\n(inference)", (o - b) * 100))
    lf = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "localize", "loc_sup0.2_bias_local_f*.npz")))
    if lf:
        parts = [np.load(f, allow_pickle=True) for f in lf]
        lab = np.concatenate([p["label"] for p in parts])
        fp = np.concatenate([p["flat_pred"] for p in parts])
        rp = np.concatenate([p["readout_pred"] for p in parts])
        costs.append(("attention bound\nto lesion", ((rp == lab).mean() - (fp == lab).mean()) * 100))
    for name, lab in (("bgdim", "background\nblurred (training)"), ("lcrop", "lesion-centred\ncrop (training)")):
        v = acc(name)
        if v is not None:
            costs.append((lab, (v - base) * 100))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    if curves:
        for lab, v in curves.items():
            ax[0].plot(MARGINS, v, "o-", lw=1.9, label=lab)
        ax[0].axhline(0, color="k", lw=0.8)
        ax[0].axhspan(-0.6, 0.6, color="#bbbbbb", alpha=0.35, lw=0)
        ax[0].text(120, 0.63, "re-training noise", ha="right", fontsize=8, color="#666")
        lim = max(1.0, np.nanmax(np.abs(np.concatenate(list(curves.values())))) * 1.35)
        ax[0].set_ylim(-lim, lim)
        ax[0].set_xlabel("context margin kept around the lesion (%)")
        ax[0].set_ylabel("accuracy change vs full image (pp)")
        ax[0].set_title("(a) How much context to keep", fontsize=11, loc="left")
        ax[0].legend(fontsize=9, frameon=False)
    if costs:
        lab = [c[0] for c in costs]; val = [c[1] for c in costs]
        o = np.argsort(val)
        ax[1].barh(np.arange(len(val)), [val[i] for i in o], color="#c0392b",
                   edgecolor="white", height=0.6)
        ax[1].set_yticks(np.arange(len(val))); ax[1].set_yticklabels([lab[i] for i in o], fontsize=9)
        ax[1].axvline(0, color="k", lw=0.8)
        ax[1].axvspan(-0.6, 0.6, color="#bbbbbb", alpha=0.35, lw=0)
        for k, i in enumerate(o):
            ax[1].text(val[i] - 0.12, k, f"{val[i]:.1f} pp", va="center", ha="right", fontsize=9, color="w")
        ax[1].set_xlabel("accuracy change (pp)")
        ax[1].set_title("(b) The cost of confining the model to the lesion", fontsize=11, loc="left")
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig4.{ext}", dpi=220, bbox_inches="tight")
    print(f" [fig4] {list(curves)} {[(l.replace(chr(10),' '), round(v,2)) for l, v in costs]}")


def loc_iou(root, name):
    fs = [os.path.join(root, "localize", f"loc_{name}_f{k}.npz") for k in range(5)]
    fs = [f for f in fs if os.path.exists(f)]
    if not fs:
        return None
    parts = [np.load(f, allow_pickle=True) for f in fs]
    R = np.concatenate([p["metrics"] for p in parts])
    nm = list(parts[0]["names"]); L = nm.index("ltv")
    ref = {n: float(np.nanmean(R[:, j, 2])) for j, n in enumerate(nm) if n != "ltv"}
    return float(np.nanmean(R[:, L, 2])), ref


def faith_val(root, name):
    fs = sorted(glob.glob(os.path.join(root, "faithful", "faith_f*_s42.npz")))
    if not fs:
        return None
    parts = [np.load(f, allow_pickle=True) for f in fs]
    if not all(f"{name}_del" in p.files for p in parts):
        return None
    d = np.concatenate([p[f"{name}_del"] for p in parts]).mean()
    i = np.concatenate([p[f"{name}_ins"] for p in parts]).mean()
    return float(i - d)


def fig5(out):
    bbs = [("MambaOut-Tiny", OUTPUT_ROOT),
           ("ConvNeXt-Tiny", os.path.join(ROOT, "outputs_convnext_tiny"))]
    rows, names = [], ["orig", "sup0.2_bias_local"]
    for lab, root in bbs:
        r0 = loc_iou(root, "orig"); r1 = loc_iou(root, "sup0.2_bias_local")
        if r0 is None or r1 is None:
            print(f"  [fig5] {lab} results "); continue
        rows.append(dict(lab=lab, unsup=r0[0], sup=r1[0], center=r0[1].get("center", np.nan),
                         gc=max(r0[1].get("gc2", 0), r0[1].get("gc4", 0)),
                         f_un=faith_val(root, "orig"), f_sup=faith_val(root, "sup0.2_bias_local"),
                         f_rnd=faith_val(root, "random")))
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    x = np.arange(len(rows)); w = 0.2
    for i, (k, lab, c) in enumerate([("unsup", "LTV (unsupervised)", "#c0392b"),
                                     ("gc", "Grad-CAM (best)", "#34495e"),
                                     ("center", "Centred Gaussian", "#b8860b"),
                                     ("sup", "LTV + mask supervision", "#1f6f3f")]):
        ax[0].bar(x + (i - 1.5) * w, [r[k] for r in rows], w, label=lab, color=c, edgecolor="white")
    ax[0].set_xticks(x); ax[0].set_xticklabels([r["lab"] for r in rows], fontsize=9)
    ax[0].set_ylabel("area-matched IoU"); ax[0].set_ylim(0, 1.0)
    ax[0].set_title("(a) Overlap with lesion masks", fontsize=11, loc="left")
    ax[0].legend(fontsize=8, frameon=False, ncol=2)
    for i, (k, lab, c) in enumerate([("f_un", "LTV (unsupervised)", "#c0392b"),
                                     ("f_rnd", "Random map", "#95a5a6"),
                                     ("f_sup", "LTV + mask supervision", "#1f6f3f")]):
        v = [r[k] if r[k] is not None else np.nan for r in rows]
        ax[1].bar(x + (i - 1) * 0.26, v, 0.26, label=lab, color=c, edgecolor="white")
    ax[1].axhline(0, color="k", lw=0.8)
    for xi, r in zip(x, rows):
        for dx, k in ((-0.26, "f_un"), (0.0, "f_rnd"), (0.26, "f_sup")):
            v = r[k]
            if v is None or np.isnan(v):
                continue
            ax[1].text(xi + dx, v + (0.006 if v >= 0 else -0.006), f"{v:+.3f}",
                       ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels([r["lab"] for r in rows], fontsize=9)
    ax[1].set_ylabel("insertion $-$ deletion")
    ax[1].set_title("(b) Faithfulness", fontsize=11, loc="left")
    ax[1].legend(fontsize=8, frameon=False)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig5.{ext}", dpi=220, bbox_inches="tight")
    print(" [fig5] " + "  ".join(f"{r['lab']}: {r['unsup']:.3f}->{r['sup']:.3f}" for r in rows))


GROUPS = [
    ("attach the module", [("d1", "gated correction"), ("d1lr", "gated correction, head lr$\\times$10"),
                           ("d1p", "gated correction, plain head"),
                           ("d3", "logit correction"), ("d3lr", "logit correction, head lr$\\times$10"),
                           ("midA", "mid-level injection (gate)"), ("midB", "mid-level injection (slots)")]),
    ("module as classifier", [("ltv", "LTV head"), ("ltvb", "LTV head, fixed masking"),
                              ("ltvn", "LTV head, norm-first masking"),
                              ("k2_bias_local", "LTV head, shortcut removed")]),
    ("training recipe", [("gap_la_ema", "logit adjustment + EMA"),
                         ("lcrop", "lesion-centred crop"), ("bgdim", "background attenuated")]),
    ("hierarchy + supervision", [("h1sup", "mask-supervised hierarchy")]),
]


def fig6(out):
    if acc("flat") is None:
        print(" [fig6] flat missing"); return
    ys, labs, vals, cols = [], [], [], []
    y = 0
    palette = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]
    for gi, (gname, items) in enumerate(GROUPS):
        got = False
        for m, lab in items:
            r = delta_vs_flat(m) or delta_vs_flat(m, sub="explore")
            if r is None:
                continue
            d, nf = r
            ys.append(y); labs.append(lab if nf == 5 else f"{lab}  ({nf}/5 folds)")
            vals.append(d); cols.append(palette[gi])
            y += 1; got = True
        if got:
            ys.append(y); labs.append(""); vals.append(np.nan); cols.append("none"); y += 1
    if not vals:
        print(" [fig6] results missing"); return
    fig, ax = plt.subplots(figsize=(7.6, 0.34 * len(vals) + 1.6))
    ax.axvspan(-0.6, 0.6, color="#bbbbbb", alpha=0.35, lw=0)
    ax.axvline(0, color="k", lw=0.9)
    ax.scatter(vals, ys, s=46, c=cols, zorder=3)
    for v, yy in zip(vals, ys):
        if not np.isnan(v):
            ax.plot([0, v], [yy, yy], color="#999", lw=0.9, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy change vs single-stage classifier (pp)")
    ax.set_title("Every design variant tested, against the re-training noise band",
                 fontsize=11, loc="left")
    ax.text(0.62, max(ys), "re-training noise", fontsize=8, color="#666", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig6.{ext}", dpi=220, bbox_inches="tight")
    print(f" [fig6] {sum(1 for v in vals if not np.isnan(v))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["4", "5", "6"])
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for w in a.which:
        {"4": fig4, "5": fig5, "6": fig6}[w](a.out)


if __name__ == "__main__":
    main()
