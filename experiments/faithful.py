"""Deletion and insertion test.

Replaces the highest-scoring positions of a saliency map by a blurred copy (deletion) or
restores them from it (insertion) and tracks the predicted probability. The difference between
the two areas under the curve is near zero for an uninformative map.

  python -m experiments.faithful --fold 0 --names orig sup0.2_bias_local
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from tqdm import tqdm

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.utils.seed import set_seed
from experiments.localize import branch_forward, center_map, feats, gradcam, load_flat, up
from experiments.localize_x import setup

OUT = os.path.join(OUTPUT_ROOT, "faithful")
os.makedirs(OUT, exist_ok=True)

def blur(x, k=51, s=20.0):
    c = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-c ** 2 / (2 * s ** 2)); g = (g / g.sum()).view(1, 1, 1, k)
    y = Fn.conv2d(Fn.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"), g.expand(3, 1, 1, k), groups=3)
    return Fn.conv2d(Fn.pad(y, (0, 0) + (k // 2,) * 2, mode="reflect"),
                     g.transpose(2, 3).expand(3, 1, k, 1), groups=3)

@torch.no_grad()
def curve(net, img, base, maps, cls, steps, mode):
    """maps [B,H,W] -> predicted probability [B, steps+1]"""
    B = img.shape[0]
    flat = maps.reshape(B, -1)
    order = flat.argsort(dim=1, descending=True)
    rank = torch.empty_like(order); ar = torch.arange(flat.shape[1], device=img.device)
    rank.scatter_(1, order, ar.expand(B, -1))
    out = []
    for i in range(steps + 1):
        n = int(flat.shape[1] * i / steps)
        sel = (rank < n).view(B, 1, IMG_SIZE, IMG_SIZE).float()
        x = img * (1 - sel) + base * sel if mode == "del" else base * (1 - sel) + img * sel
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            p = torch.softmax(net.forward_head(net.forward_intermediates(x, intermediates_only=False)[0]).float(), 1)
        out.append(p.gather(1, cls[:, None]).squeeze(1).cpu())
    return torch.stack(out, 1).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--names", nargs="+", default=["orig", "bias_local", "k2_bias_local"])
    ap.add_argument("--n", type=int, default=400, help="fold")
    ap.add_argument("--steps", type=int, default=10)
    a = ap.parse_args()
    set_seed(a.seed)
    from experiments.residual import get_data
    _, va, _ = get_data(a.fold, build_image_path_map())
    va = va.head(a.n).reset_index(drop=True)
    net = load_flat(a.fold)
    loc = os.path.join(OUTPUT_ROOT, "localize") + ("" if a.seed == 42 else f"/seed{a.seed}")

    brs = {}
    for n in a.names:
        ck = os.path.join(loc, f"readout_{n}_f{a.fold}.pth")
        if not os.path.exists(ck):
            print(f"  [{n}] readout missing, "); continue
        import experiments.localize as L
        if n.startswith("sup"):
            from experiments.localize_sup import build as build_sup
            L.feats = feats
            br, s_dim = build_sup(net, mode="bias", local=not n.startswith("supg"))
        else:
            setup(n)
            br, s_dim = L.make_branch(net, n)
        br.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True), strict=True)
        br.eval(); brs[n] = (br, s_dim, L.feats)
    if not brs:
        print("measurement readout missing"); return
    keys = list(brs) + ["gc4", "gc2", "center", "random"]
    rng = np.random.default_rng(a.seed)
    cen = torch.tensor(center_map(), dtype=torch.float32, device=DEVICE)
    res = {k: {"del": [], "ins": []} for k in keys}
    labels, preds = [], []

    for img, lbl in tqdm(make_loader(va, "label", build_val_transform(), shuffle=False), leave=False):
        img = img.to(DEVICE).requires_grad_(True)
        import experiments.localize as L
        L.feats = feats
        store = {}
        h = net.stages[1].register_forward_hook(lambda m, i, o: (o.retain_grad(), store.__setitem__("s2", o))[1])
        with torch.enable_grad():
            f, s2, _ = feats(net, img)
            f.retain_grad()
            logits = net.forward_head(f)
            cls = logits.argmax(1)
            logits.gather(1, cls[:, None]).sum().backward()
            g4 = gradcam(f.detach(), f.grad, nhwc=True)
            h2 = store["s2"]; g2 = gradcam(h2.detach(), h2.grad, nhwc=(h2.shape[-1] != h2.shape[-2]))
        h.remove(); net.zero_grad(set_to_none=True)
        x = img.detach(); base = blur(x)
        M = {"gc4": up(g4).detach(), "gc2": up(g2).detach(),
             "center": cen.expand(x.shape[0], -1, -1),
             "random": torch.tensor(rng.random((x.shape[0], IMG_SIZE, IMG_SIZE)), dtype=torch.float32, device=DEVICE)}
        with torch.no_grad():
            for n, (br, s_dim, ft) in brs.items():
                if not n.startswith("sup"):
                    setup(n)
                else:
                    L.feats = feats
                ff, ss, _ = L.feats(net, x)
                _, sal = branch_forward(br, s_dim, net.forward_head(ff, pre_logits=True), ss)
                imp = sal.mean(1).max(1).values
                gsz = int(round(imp.shape[1] ** 0.5))
                M[n] = up(imp.view(-1, gsz, gsz)).detach()
            for k in keys:
                res[k]["del"].append(curve(net, x, base, M[k], cls, a.steps, "del"))
                res[k]["ins"].append(curve(net, x, base, M[k], cls, a.steps, "ins"))
        labels.append(lbl.numpy()); preds.append(cls.cpu().numpy())

    tag = f"f{a.fold}_s{a.seed}"
    D = {f"{k}_{m}": np.concatenate(res[k][m]) for k in keys for m in ("del", "ins")}
    np.savez(os.path.join(OUT, f"faith_{tag}.npz"), label=np.concatenate(labels),
             pred=np.concatenate(preds), names=np.array(keys), **D)
    print(f"[faithful:{tag}] n={len(np.concatenate(labels))}")
    print(f"  {'supervised':<16}{'deletion AUC':>14}{'insertion AUC':>15} (deletion , insertion )")
    for k in keys:
        d = np.concatenate(res[k]["del"]).mean(1); i = np.concatenate(res[k]["ins"]).mean(1)
        print(f"  {k:<16}{d.mean():>14.4f}{i.mean():>15.4f}")

if __name__ == "__main__":
    main()
