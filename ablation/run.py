"""Training and evaluation entry point for the seed-42 ablation."""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from SkinCancer.configs.data_config import (
    FINAL_CLASS_NAMES, NM_CLASS_NAMES, STAGE2_MELANOCYTIC_MAPPING,
    STAGE2_NONMELANOCYTIC_MAPPING,
)
from SkinCancer.configs.training_config import (
    ACCUMULATION_STEPS, BOOTSTRAP_N, DEVICE, EPOCHS_STAGE2_MELANOCYTIC,
    EPOCHS_STAGE2_NONMELANOCYTIC, LEARNING_RATE, OUTPUT_ROOT, SEED,
    STAGE1_MODEL_PATH, WEIGHT_DECAY,
)
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import (
    build_image_path_map, load_final7_test_data,
    load_stage2_melanocytic_data, load_stage2_nonmelanocytic_data,
)
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.eval.bootstrap import bootstrap_macro_ci
from SkinCancer.eval.evaluate import validate
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics
from SkinCancer.models.stage1 import create_stage1_classifier
from SkinCancer.train.train_utils import train_one_epoch
from SkinCancer.utils.seed import set_seed
from ablation.models import build, describe


def ckpt_path(tag, branch):
    return os.path.join(OUTPUT_ROOT, f"abl_{tag}_stage2_{branch}.pth")


def train_branch(branch, tag, teme, ltv, simple):
    path_map = build_image_path_map()
    if branch == "mel":
        tr, va = load_stage2_melanocytic_data(path_map)
        n_cls, epochs, multiclass = 1, EPOCHS_STAGE2_MELANOCYTIC, False
        n_pos = int((tr["stage2_label"] == 1).sum())
        n_neg = int((tr["stage2_label"] == 0).sum())
        pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"[abl:{tag}][mel] train={len(tr)} val={len(va)} pos_weight={pw.item():.4f}")
    else:
        tr, va = load_stage2_nonmelanocytic_data(path_map)
        n_cls, epochs, multiclass = 5, EPOCHS_STAGE2_NONMELANOCYTIC, True
        counts = tr["stage2_label"].value_counts().sort_index().values
        w = torch.tensor(counts.sum() / (len(counts) * counts),
                         dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(weight=w)
        print(f"[abl:{tag}][nonmel] train={len(tr)} val={len(va)} weights="
              + ", ".join(f"{c}:{v:.3f}" for c, v in zip(NM_CLASS_NAMES, w.tolist())))

    model = build(n_cls, teme, ltv, simple).to(DEVICE)
    print(f"[abl:{tag}] {describe(model)}")

    tl = make_loader(tr, "stage2_label", build_train_transform(), shuffle=True)
    vl = make_loader(va, "stage2_label", build_val_transform(), shuffle=False)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    out = ckpt_path(tag, branch)
    best = -1.0
    for ep in range(epochs):
        loss = train_one_epoch(model, tl, criterion, opt, scaler, is_multiclass=multiclass,
                               epoch=ep + 1, total_epochs=epochs, stage_name=f"abl:{tag}:{branch}")
        auc = validate(model, vl, criterion, is_multiclass=multiclass,
                       stage_name=f"abl:{tag}:{branch}", epoch=ep + 1, total_epochs=epochs,
                       class_names=NM_CLASS_NAMES if multiclass else None)
        print(f"Epoch {ep+1:02d}/{epochs} | Loss: {loss:.4f} | Val AUC: {auc:.4f}")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out)
            print(f"  -> checkpoint saved (val AUC {best:.4f})")
    print(f"[abl:{tag}][{branch}] best val AUC = {best:.4f} -> {out}")


@torch.no_grad()
def route(tag, teme, ltv, simple, routing, threshold):
    path_map = build_image_path_map()
    te = load_final7_test_data(path_map)
    loader = make_loader(te, "final_label", build_val_transform(), shuffle=False)

    s1 = create_stage1_classifier().to(DEVICE)
    s1.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE, weights_only=True),
                       strict=True)
    mel = build(1, teme, ltv, simple).to(DEVICE)
    mel.load_state_dict(torch.load(ckpt_path(tag, "mel"), map_location=DEVICE,
                                   weights_only=True), strict=True)
    nm = build(5, teme, ltv, simple).to(DEVICE)
    nm.load_state_dict(torch.load(ckpt_path(tag, "nonmel"), map_location=DEVICE,
                                  weights_only=True), strict=True)
    for m in (s1, mel, nm):
        m.eval()

    probs, trues = [], []
    for img, lbl in tqdm(loader, desc=f"abl:{tag}:{routing}", leave=False):
        img = img.to(DEVICE)
        out = torch.zeros(img.shape[0], 7, device=DEVICE, dtype=torch.float32)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            p1 = torch.sigmoid(s1(img)).view(-1, 1).float()
            ml = mel(img)
            nl = nm(img)
        p_mel = torch.sigmoid(ml).view(-1, 1).float()
        mel_p = torch.cat([1.0 - p_mel, p_mel], dim=1)
        nm_p = torch.softmax(nl, dim=1).float()

        gate = (p1 > threshold).float() if routing == "hard" else p1
        out[:, 0:2] = gate * mel_p
        out[:, 2:7] = (1.0 - gate) * nm_p
        probs.extend(out.cpu().numpy())
        trues.extend(lbl.numpy().flatten())

    y_prob, y_true = np.array(probs), np.array(trues).astype(int)
    suffix = f"{tag}_{routing}"
    np.savez(os.path.join(OUTPUT_ROOT, f"abl_{suffix}_predictions.npz"),
             y_prob=y_prob, y_true=y_true, y_pred=y_prob.argmax(1),
             image_id=te["image_id"].values)

    overall, per_class = calculate_detailed_multiclass_metrics(y_true, y_prob, FINAL_CLASS_NAMES)
    print("\n" + "=" * 66)
    print(f"   [{tag}]  routing={routing}  TEME={int(teme)} LTV={int(ltv)} simple={int(simple)}")
    print("=" * 66)
    print("   [Overall Macro Metrics]")
    for k in ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1", "PPV", "NPV"]:
        print(f"   {k:<14} | {overall[k]:.4f}")
    print("\n   [Per-Class]")
    for c in FINAL_CLASS_NAMES:
        m = per_class[c]
        print(f"   {c.upper():<6} prec={m['PPV']*100:6.2f}%  rec={m['Sensitivity']*100:6.2f}%  "
              f"f1={m['F1']*100:6.2f}%  n={m['Support']}")

    ci = bootstrap_macro_ci(y_true, y_prob, FINAL_CLASS_NAMES, n_boot=BOOTSTRAP_N, seed=SEED)
    ci.to_csv(os.path.join(OUTPUT_ROOT, f"abl_{suffix}_macro_95ci.csv"), index=False)
    print(f"\n : {OUTPUT_ROOT}/abl_{suffix}_predictions.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["mel", "nonmel", "routing"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--teme", type=int, default=0)
    ap.add_argument("--ltv", type=int, default=1)
    ap.add_argument("--simple-head", type=int, default=0)
    ap.add_argument("--routing", default="soft", choices=["soft", "hard"])
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    set_seed(SEED)
    if args.stage == "routing":
        route(args.tag, args.teme, args.ltv, args.simple_head, args.routing, args.threshold)
    else:
        train_branch(args.stage, args.tag, args.teme, args.ltv, args.simple_head)


if __name__ == "__main__":
    main()
