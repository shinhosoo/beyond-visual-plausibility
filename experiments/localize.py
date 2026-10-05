"""Measure how well a saliency map overlaps the lesion.

Takes the single-stage classifier of a fold as a frozen backbone, attaches a TEME+LTV readout,
and compares several maps against the official lesion masks: the LTV attention, Grad-CAM at the
final and at the second stage, a centred Gaussian and a random map.

  python -m experiments.localize --stage train --fold 0 --ltv-mode orig
  python -m experiments.localize --stage eval  --fold 0 --ltv-mode orig
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE, OUTPUT_ROOT
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.models.teme import smart_reshape
from SkinCancer.utils.seed import set_seed
from experiments.residual import ResidualModel, get_data, inv_freq
from experiments.shared import Branch

OUT = os.path.join(OUTPUT_ROOT, "localize")
os.makedirs(OUT, exist_ok=True)
MASK_DIR = os.environ.get("HAM_MASK_DIR", "masks_ham10000")


def load_flat(fold):
    m = ResidualModel("flat", pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUTPUT_ROOT, "residual", f"flat_f{fold}_s42.pth"),
                                 map_location=DEVICE, weights_only=True), strict=True)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.net


def feats(net, x):
    f, st = net.forward_intermediates(x, intermediates_only=False)
    s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
    return f, s2, st


def make_branch(net, mode):
    with torch.no_grad():
        f, s2, _ = feats(net, torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))
        s2_dim = s2.shape[1]
        final_dim = net.forward_head(f, pre_logits=True).shape[1]
    br = Branch(s2_dim, final_dim, 7, teme=True, ltv=True)
    if mode != "orig":
        from experiments.ltv_fix import LTVFixed
        from SkinCancer.configs.model_config import D_ATTN, N_HEADS, N_SLOTS, SLOT_AGG
        br.ltv_fusion = LTVFixed(d_macro=final_dim, d_micro=256, d_attn=D_ATTN, n_heads=N_HEADS,
                                 n_slots=N_SLOTS, dropout=0.1, slot_agg=SLOT_AGG, mode=mode)
    return br.to(DEVICE), s2_dim


def branch_forward(br, s2_dim, macro, s2):
    micro = br.micro_proj(smart_reshape(s2, s2_dim))
    fused, sal = br.ltv_fusion(macro, micro)
    return br.classifier(fused), sal


def train(a):
    set_seed(a.seed)
    tr, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        tr = tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True)
    net = load_flat(a.fold)
    br, s2_dim = make_branch(net, a.ltv_mode)
    y = tr["label"].astype(int).values
    ce = nn.CrossEntropyLoss(weight=inv_freq(np.bincount(y, minlength=7)))
    opt = torch.optim.AdamW(br.parameters(), lr=5e-4, weight_decay=5e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = make_loader(tr, "label", build_train_transform(), shuffle=True)
    vl = make_loader(va, "label", build_val_transform(), shuffle=False)
    tag = f"{a.ltv_mode}_f{a.fold}"
    print(f"[localize:{tag}] backbone readout training train={len(tr)} val={len(va)}  "
          f"params={sum(p.numel() for p in br.parameters()):,}")
    best, ck = -1.0, os.path.join(OUT, f"readout_{tag}.pth")
    for ep in range(a.epochs):
        br.train(); tot = 0.0
        for img, lbl in tl:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE).long()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                f, s2, _ = feats(net, img)
                macro = net.forward_head(f, pre_logits=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                z, _ = branch_forward(br, s2_dim, macro.float(), s2.float())
                loss = ce(z, lbl)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss)
        br.eval(); P, Y = [], []
        with torch.no_grad():
            for img, lbl in vl:
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    f, s2, _ = feats(net, img.to(DEVICE))
                    z, _ = branch_forward(br, s2_dim, net.forward_head(f, pre_logits=True).float(), s2.float())
                P.append(torch.softmax(z.float(), 1).cpu().numpy()); Y.append(lbl.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y).astype(int)
        try:
            auc = roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr", average="macro")
        except Exception:
            auc = float((P.argmax(1) == Y).mean())
        print(f"Epoch {ep+1:02d}/{a.epochs} | Loss {tot/max(len(tl),1):.4f} | readout Val AUC {auc:.4f} "
              f"Acc {(P.argmax(1)==Y).mean():.4f}")
        if auc > best:
            best = auc; torch.save(br.state_dict(), ck)
    print(f"[localize:{tag}] best readout val AUC {best:.4f} -> {ck}")


def gradcam(act, grad, nhwc):
    if nhwc:
        w = grad.mean(dim=(1, 2)); cam = (act * w[:, None, None, :]).sum(-1)
    else:
        w = grad.mean(dim=(2, 3)); cam = (act * w[:, :, None, None]).sum(1)
    return Fn.relu(cam)


def up(m):
    return Fn.interpolate(m[:, None].float(), size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False)[:, 0]


def metrics(maps, mask):
    out = []
    area = mask.mean()
    for m in maps:
        m = m - m.min()
        flat = m.reshape(-1)
        point = float(mask.reshape(-1)[flat.argmax()])
        energy = float((m * mask).sum() / max(m.sum(), 1e-12))
        if area <= 0 or area >= 1:
            iou = np.nan
        else:
            thr = np.quantile(flat, 1 - area)
            sel = m >= thr
            iou = float((sel & mask).sum() / max((sel | mask).sum(), 1))
        out.append((point, energy, iou))
    return np.array(out)


def center_map():
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    c, s = (IMG_SIZE - 1) / 2, IMG_SIZE * 0.25
    return np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * s ** 2))


def evaluate(a):
    _, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        va = va.head(a.limit).reset_index(drop=True)
    net = load_flat(a.fold)
    br, s2_dim = make_branch(net, a.ltv_mode)
    tag = f"{a.ltv_mode}_f{a.fold}"
    br.load_state_dict(torch.load(os.path.join(OUT, f"readout_{tag}.pth"), map_location=DEVICE,
                                  weights_only=True), strict=True)
    br.eval()
    cen = center_map()
    names = ["ltv", "gc4", "gc2", "center"]
    rows, ids, labels, fpred, rpred, areas = [], [], [], [], [], []
    loader = make_loader(va, "label", build_val_transform(), shuffle=False)
    store = {}
    def _hook(mod, inp, out):
        out.retain_grad(); store["s2"] = out
    hook = net.stages[1].register_forward_hook(_hook)
    k = 0
    for img, lbl in tqdm(loader, desc=f"eval {tag}", leave=False):
        img = img.to(DEVICE).requires_grad_(True)
        with torch.enable_grad():
            f, s2, st = feats(net, img)
            f.retain_grad()
            logits = net.forward_head(f)
            pred = logits.argmax(1)
            logits.gather(1, pred[:, None]).sum().backward()
            gc4 = gradcam(f.detach(), f.grad, nhwc=True)
            h2 = store["s2"]
            gc2 = gradcam(h2.detach(), h2.grad, nhwc=(h2.shape[-1] != h2.shape[-2]))
        with torch.no_grad():
            z, sal = branch_forward(br, s2_dim, net.forward_head(f.detach(), pre_logits=True), s2.detach())
            imp = sal.mean(1).max(1).values
            g = int(round(imp.shape[1] ** 0.5))
            ltv = imp.view(-1, g, g)
        L, G4, G2 = up(ltv).cpu().numpy(), up(gc4).cpu().numpy(), up(gc2).cpu().numpy()
        for i in range(img.shape[0]):
            iid = va.iloc[k]["image_id"]; k += 1
            mp = os.path.join(MASK_DIR, f"{iid}_segmentation.png")
            mask = np.array(Image.open(mp).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)) > 127
            rows.append(metrics(np.stack([L[i], G4[i], G2[i], cen]), mask))
            ids.append(iid); labels.append(int(lbl[i])); fpred.append(int(pred[i]))
            rpred.append(int(z[i].argmax())); areas.append(float(mask.mean()))
        net.zero_grad(set_to_none=True)
    hook.remove()
    R = np.stack(rows)
    f_ = os.path.join(OUT, f"loc_{tag}.npz")
    np.savez(f_, metrics=R, names=np.array(names), image_id=np.array(ids), label=np.array(labels),
             flat_pred=np.array(fpred), readout_pred=np.array(rpred), lesion_area=np.array(areas))
    print(f"[localize:{tag}] n={len(ids)}  flat acc={np.mean(np.array(fpred)==np.array(labels)):.4f}  "
          f"readout acc={np.mean(np.array(rpred)==np.array(labels)):.4f}")
    for j, n in enumerate(names):
        print(f"  {n:<7} pointing {np.nanmean(R[:,j,0]):.3f}  energy {np.nanmean(R[:,j,1]):.3f}  "
              f"IoU {np.nanmean(R[:,j,2]):.3f}")
    print(f" : {f_}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ltv-mode", default="orig", choices=["orig", "bias", "norm"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    (train if a.stage == "train" else evaluate)(a)


if __name__ == "__main__":
    main()
