"""Qualitative figure: image, lesion mask, unsupervised LTV, mask-supervised LTV, Grad-CAM."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed
from experiments.localize import branch_forward, feats, gradcam, load_flat, make_branch, up
from experiments.localize_sup import build as build_sup
from experiments.residual import get_data

MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")
CLS = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]

def norm01(m):
    m = m - m.min()
    return m / max(m.max(), 1e-12)

def overlay(ax, img, m=None, mask=None, title=""):
    ax.imshow(img)
    if m is not None:
        ax.imshow(norm01(m), cmap="jet", alpha=0.45)
    if mask is not None:
        ax.contour(mask.astype(float), levels=[0.5], colors="w", linewidths=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sup", default="sup0.2_bias_local")
    ap.add_argument("--n", type=int, default=200, help=" ")
    ap.add_argument("--out", default="figs/fig3")
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    _, va, _ = get_data(a.fold, build_image_path_map())
    va = va.head(a.n).reset_index(drop=True)

    net = load_flat(a.fold)
    un, s2d = make_branch(net, "orig")
    un.load_state_dict(torch.load(os.path.join(OUTPUT_ROOT, "localize", f"readout_orig_f{a.fold}.pth"),
                                  map_location=DEVICE, weights_only=True), strict=True); un.eval()
    sp, _ = build_sup(net, mode="bias", local=True)
    sp.load_state_dict(torch.load(os.path.join(OUTPUT_ROOT, "localize", f"readout_{a.sup}_f{a.fold}.pth"),
                                  map_location=DEVICE, weights_only=True), strict=True); sp.eval()

    from torchvision import transforms
    from torchvision.transforms import functional as TF
    NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    rec = []
    for i in range(len(va)):
        r = va.iloc[i]
        im = Image.open(r["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mk = np.array(Image.open(os.path.join(MASK_DIR, f"{r['image_id']}_segmentation.png"))
                      .convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
        x = NORM(TF.to_tensor(im))[None].to(DEVICE).requires_grad_(True)
        store = {}
        h = net.stages[1].register_forward_hook(lambda m, i_, o: (o.retain_grad(), store.__setitem__("s2", o))[1])
        with torch.enable_grad():
            f, s2, _ = feats(net, x)
            f.retain_grad()
            z = net.forward_head(f); pred = int(z.argmax(1))
            z[0, pred].backward()
            h2 = store["s2"]
            gc = up(gradcam(h2.detach(), h2.grad, nhwc=(h2.shape[-1] != h2.shape[-2])))[0].cpu().numpy()
        h.remove(); net.zero_grad(set_to_none=True)
        maps = {}
        with torch.no_grad():
            for key, br in (("unsup", un), ("sup", sp)):
                _, sal = branch_forward(br, s2d, net.forward_head(f.detach(), pre_logits=True), s2.detach())
                imp = sal.mean(1).max(1).values
                g = int(round(imp.shape[1] ** 0.5))
                maps[key] = up(imp.view(-1, g, g))[0].cpu().numpy()
        maps["gc"] = gc
        area = mk.mean()
        yy, xx = np.where(mk)
        cy, cx = (yy.mean(), xx.mean()) if len(yy) else (IMG_SIZE / 2, IMG_SIZE / 2)
        off = np.hypot(cy - IMG_SIZE / 2, cx - IMG_SIZE / 2) / IMG_SIZE
        m = norm01(maps["unsup"]); thr = np.quantile(m, 1 - area)
        iou = ((m >= thr) & mk).sum() / max(((m >= thr) | mk).sum(), 1)
        conc = float((m >= np.quantile(m, 0.99)).mean())
        peak = float(m.max() / max(m.mean(), 1e-9))
        rec.append(dict(i=i, id=r["image_id"], label=int(r["label"]), pred=pred, area=area,
                        off=off, iou=iou, peak=peak, img=np.asarray(im), mask=mk, **maps))

    df = rec
    mid = [d for d in df if 0.25 <= d["area"] <= 0.55 and d["off"] < 0.08]
    big = max(mid or df, key=lambda d: -abs(d["area"] - 0.4))
    sm = [d for d in df if d["area"] < 0.12 and d["off"] > 0.12]
    small = max(sm or df, key=lambda d: d["off"])
    used = {big["id"], small["id"]}
    cand = [d for d in df if d["id"] not in used and d["area"] >= 0.2]
    cand = sorted(cand, key=lambda d: -d["peak"])[:max(5, len(cand) // 5)]
    plaus = min(cand or df, key=lambda d: d["iou"])
    picks = [("(a) large, central lesion", big),
             ("(b) small, off-centre lesion", small),
             ("(c) large lesion, attention elsewhere", plaus)]
    used |= {plaus["id"]}
    wrong = [d for d in df if d["pred"] != d["label"] and d["id"] not in used and d["area"] >= 0.15]
    if wrong:
        picks.append(("(d) misclassified case", max(wrong, key=lambda d: d["area"])))

    cols = ["image", "lesion mask", "LTV (unsupervised)", "LTV + mask supervision", "Grad-CAM"]
    fig, axes = plt.subplots(len(picks), 5, figsize=(13, 2.75 * len(picks)))
    for r_, (cap, d) in enumerate(picks):
        overlay(axes[r_, 0], d["img"], title=cols[0] if r_ == 0 else "")
        overlay(axes[r_, 1], d["img"], mask=d["mask"], title=cols[1] if r_ == 0 else "")
        for c_, key in enumerate(["unsup", "sup", "gc"], start=2):
            overlay(axes[r_, c_], d["img"], m=d[key], mask=d["mask"],
                    title=cols[c_] if r_ == 0 else "")
        axes[r_, 0].set_ylabel(cap, fontsize=9)
        axes[r_, 0].text(4, 22, f"{CLS[d['label']]} -> {CLS[d['pred']]}", color="w", fontsize=8,
                         bbox=dict(fc="k", alpha=0.45, pad=1.5, lw=0))
        axes[r_, 2].text(4, 22, f"IoU {d['iou']:.2f}", color="w", fontsize=8,
                         bbox=dict(fc="k", alpha=0.45, pad=1.5, lw=0))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f": {a.out}.pdf / .png")
    for cap, d in picks:
        print(f"  {cap}: {d['id']} {d['area']:.2f} {d['off']:.2f} unsupervised IoU {d['iou']:.2f}")

if __name__ == "__main__":
    main()
