"""Two qualitative figures: all explanation maps side by side, and the deletion test."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.utils.seed import set_seed
from experiments.localize import branch_forward, feats, gradcam, load_flat, make_branch, up
from experiments.localize_sup import build as build_sup
from experiments.residual import get_data

MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")
CLS = ["NV", "MEL", "BKL", "DF", "VASC", "BCC", "AKIEC"]
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def n01(m):
    m = m - m.min()
    return m / max(m.max(), 1e-12)


def center_map(size=IMG_SIZE, sigma=0.25):
    y, x = np.mgrid[0:size, 0:size] / size - 0.5
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def load_case(row, net, un, sp, s2d):
    im = Image.open(row["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    mk = np.array(Image.open(os.path.join(MASK_DIR, f"{row['image_id']}_segmentation.png"))
                  .convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
    x = NORM(TF.to_tensor(im))[None].to(DEVICE).requires_grad_(True)
    store = {}
    h = net.stages[1].register_forward_hook(
        lambda m, i_, o: (o.retain_grad(), store.__setitem__("s2", o))[1])
    with torch.enable_grad():
        f, s2, _ = feats(net, x)
        z = net.forward_head(f)
        pred = int(z.argmax(1)); prob = float(torch.softmax(z, 1)[0, pred])
        z[0, pred].backward()
        h2 = store["s2"]
        gc = up(gradcam(h2.detach(), h2.grad, nhwc=(h2.shape[-1] != h2.shape[-2])))[0].cpu().numpy()
    h.remove(); net.zero_grad(set_to_none=True)
    maps = {"gc": gc}
    with torch.no_grad():
        for key, br in (("unsup", un), ("sup", sp)):
            _, sal = branch_forward(br, s2d, net.forward_head(f.detach(), pre_logits=True), s2.detach())
            imp = sal.mean(1).max(1).values
            g = int(round(imp.shape[1] ** 0.5))
            maps[key] = up(imp.view(-1, g, g))[0].cpu().numpy()
    maps["center"] = center_map()
    maps["random"] = np.random.default_rng(0).random((IMG_SIZE, IMG_SIZE))
    return dict(img=np.asarray(im), mask=mk, pred=pred, prob=prob,
                label=int(row["label"]), id=row["image_id"], **maps)


def iou_of(m, mask):
    m = n01(m); thr = np.quantile(m, 1 - mask.mean())
    sel = m >= thr
    return float((sel & mask).sum() / max((sel | mask).sum(), 1))


def fig_A(cases, out):
    cols = [("img", "image"), ("mask", "lesion mask"), ("unsup", "LTV (unsupervised)"),
            ("sup", "LTV + mask supervision"), ("gc", "Grad-CAM"),
            ("center", "centred Gaussian"), ("random", "random map")]
    fig, ax = plt.subplots(len(cases), len(cols), figsize=(2.05 * len(cols), 2.2 * len(cases)))
    ax = np.atleast_2d(ax)
    for r, d in enumerate(cases):
        for c, (k, lab) in enumerate(cols):
            a = ax[r, c]
            a.imshow(d["img"])
            if k not in ("img", "mask"):
                a.imshow(n01(d[k]), cmap="jet", alpha=0.45)
                a.text(4, 22, f"IoU {iou_of(d[k], d['mask']):.2f}", color="w", fontsize=7.5,
                       bbox=dict(fc="k", alpha=0.45, pad=1.2, lw=0))
            if k != "img":
                a.contour(d["mask"].astype(float), levels=[0.5], colors="w", linewidths=1.0)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(lab, fontsize=9)
        ax[r, 0].set_ylabel(f"{CLS[d['label']]} $\\rightarrow$ {CLS[d['pred']]}", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig_maps.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" fig_maps ({len(cases)} × {len(cols)})")


@torch.no_grad()
def curve_images(net, d, key, steps=(0.0, 0.2, 0.4, 0.6, 0.8), mode="del"):
    img = torch.tensor(d["img"] / 255.0).permute(2, 0, 1).float()
    blur = TF.gaussian_blur(img, kernel_size=51, sigma=20.0)
    m = torch.tensor(n01(d[key])).flatten()
    order = torch.argsort(m, descending=True)
    outs = []
    for t in steps:
        k = int(t * len(order))
        keep = torch.ones(len(order))
        keep[order[:k]] = 0.0
        keep = keep.view(1, IMG_SIZE, IMG_SIZE)
        x = img * keep + blur * (1 - keep) if mode == "del" else blur * keep + img * (1 - keep)
        xi = NORM(x)[None].to(DEVICE)
        f, _, _ = feats(net, xi)
        p = float(torch.softmax(net.forward_head(f).float(), 1)[0, d["pred"]])
        outs.append((x.permute(1, 2, 0).numpy(), p, t))
    return outs


def fig_B(net, d, out):
    keys = [("sup", "LTV + mask supervision"), ("gc", "Grad-CAM"), ("unsup", "LTV (unsupervised)"),
            ("random", "random map")]
    steps = (0.0, 0.2, 0.4, 0.6, 0.8)
    fig, ax = plt.subplots(len(keys), len(steps) + 1, figsize=(2.0 * (len(steps) + 1), 2.15 * len(keys)))
    for r, (k, lab) in enumerate(keys):
        a0 = ax[r, 0]
        a0.imshow(d["img"]); a0.imshow(n01(d[k]), cmap="jet", alpha=0.45)
        a0.contour(d["mask"].astype(float), levels=[0.5], colors="w", linewidths=1.0)
        a0.set_xticks([]); a0.set_yticks([]); a0.set_ylabel(lab, fontsize=8.5)
        if r == 0:
            a0.set_title("saliency", fontsize=9)
        for c, (im, p, t) in enumerate(curve_images(net, d, k, steps), start=1):
            a = ax[r, c]
            a.imshow(np.clip(im, 0, 1)); a.set_xticks([]); a.set_yticks([])
            a.set_xlabel(f"p = {p:.2f}", fontsize=8.5)
            if r == 0:
                a.set_title(f"{int(t*100)}\\% removed", fontsize=9)
    fig.suptitle(f"Deletion test on one image (predicted {CLS[d['pred']]}, "
                 f"initial p = {d['prob']:.2f})", fontsize=10, y=1.003)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/fig_deletion.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(" fig_deletion ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["A", "B"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sup", default="sup0.2_bias_local")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    _, va, _ = get_data(a.fold, build_image_path_map())
    va = va.head(a.n).reset_index(drop=True)
    net = load_flat(a.fold)
    un, s2d = make_branch(net, "orig")
    un.load_state_dict(torch.load(os.path.join(OUTPUT_ROOT, "localize", f"readout_orig_f{a.fold}.pth"),
                                  map_location=DEVICE, weights_only=True), strict=True); un.eval()
    sp, _ = build_sup(net, mode="bias", local=True)
    sp.load_state_dict(torch.load(os.path.join(OUTPUT_ROOT, "localize", f"readout_{a.sup}_f{a.fold}.pth"),
                                  map_location=DEVICE, weights_only=True), strict=True); sp.eval()

    rec = []
    for i in range(len(va)):
        d = load_case(va.iloc[i], net, un, sp, s2d)
        d["area"] = d["mask"].mean()
        rec.append(d)
    ok = [d for d in rec if d["pred"] == d["label"]]
    mid = [d for d in ok if 0.2 <= d["area"] <= 0.6]
    pick = sorted(mid or ok, key=lambda d: -d["prob"])[:3]

    if "A" in a.which:
        fig_A(pick, a.out)
        for d in pick:
            print(f"    {d['id']}  {CLS[d['label']]} {d['area']:.2f}  "
                  f"unsupervised IoU {iou_of(d['unsup'], d['mask']):.2f}  "
                  f" IoU {iou_of(d['random'], d['mask']):.2f}")
    if "B" in a.which:
        fig_B(net, pick[0], a.out)


if __name__ == "__main__":
    main()
