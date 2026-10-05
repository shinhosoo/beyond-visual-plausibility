"""LTV used as the classification head, with optional training recipes.

Tags combine a base and optional suffixes: gap (single-stage), ltv (GAP replaced by TEME+LTV),
_la (logit adjustment), _ema (exponential moving average of the weights).
"""
import argparse, os
import numpy as np, timm, torch, torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from SkinCancer.configs.model_config import BACKBONE_NAME, IMG_SIZE
from SkinCancer.configs.training_config import ACCUMULATION_STEPS, DEVICE, EPOCHS_STAGE1, LEARNING_RATE, OUTPUT_ROOT, WEIGHT_DECAY
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.utils.seed import set_seed
from experiments.residual import get_data, inv_freq, macro_auc
from experiments.shared import Branch

OUT = os.path.join(OUTPUT_ROOT, "explore"); os.makedirs(OUT, exist_ok=True)


def parse(name):
    parts = name.split("_"); assert parts[0] in ("gap", "ltv"), name
    return parts[0], "la" in parts[1:], "ema" in parts[1:]


class ExpModel(nn.Module):
    def __init__(self, head, pretrained=True):
        super().__init__()
        self.head = head
        self.net = timm.create_model(BACKBONE_NAME, pretrained=pretrained, num_classes=7)
        if hasattr(self.net, "set_grad_checkpointing"):
            self.net.set_grad_checkpointing(True)
        if head == "ltv":
            with torch.no_grad():
                f, st = self.net.forward_intermediates(torch.randn(1, 3, IMG_SIZE, IMG_SIZE), intermediates_only=False)
                s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
                s2_dim = s2.shape[1] if s2.dim() == 4 else s2.shape[-1]
                final_dim = self.net.forward_head(f, pre_logits=True).shape[1]
            self.br = Branch(s2_dim, final_dim, 7, teme=True, ltv=True)

    def forward(self, x):
        if self.head == "gap":
            return self.net(x)
        f, st = self.net.forward_intermediates(x, intermediates_only=False)
        s2 = st[1][0] if isinstance(st[1], (list, tuple)) else st[1]
        return self.br(self.net.forward_head(f, pre_logits=True), s2)


@torch.no_grad()
def infer(model, loader):
    model.eval(); P, Y = [], []
    for img, lbl in loader:
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            z = model(img.to(DEVICE))
        P.append(torch.softmax(z.float(), 1).cpu().numpy()); Y.append(lbl.numpy().flatten())
    return np.concatenate(P), np.concatenate(Y).astype(int)


def run(a):
    set_seed(a.seed)
    head, use_la, use_ema = parse(a.name)
    tag = f"{a.name}_f{a.fold}_s{a.seed}"
    tr, va, _ = get_data(a.fold, build_image_path_map())
    if a.limit:
        tr = tr.groupby("label").head(a.limit // 7 + 1).reset_index(drop=True)
    y = tr["label"].astype(int).values
    counts = np.maximum(np.bincount(y, minlength=7), 1)
    if use_la:
        adj = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32, device=DEVICE)
        ce = nn.CrossEntropyLoss(); crit = lambda z, t: ce(z + adj, t)
    else:
        ce = nn.CrossEntropyLoss(weight=inv_freq(counts)); crit = lambda z, t: ce(z, t)
    print(f"[explore:{tag}] head={head} logit_adjust={use_la} ema={use_ema} train={len(tr)} val={len(va)}")
    model = ExpModel(head, pretrained=not a.no_pretrained).to(DEVICE)
    bb = [p for n, p in model.named_parameters() if n.startswith("net.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("net.")]
    groups = [{"params": bb, "lr": LEARNING_RATE}] + ([{"params": hd, "lr": LEARNING_RATE * 10}] if hd else [])
    opt = torch.optim.AdamW(groups, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"[explore:{tag}] lr backbone={LEARNING_RATE:g}  heads={LEARNING_RATE*10:g} ({len(hd)} tensors)")
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.998), use_buffers=True) if use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    tl = make_loader(tr, "label", build_train_transform(), shuffle=True)
    vl = make_loader(va, "label", build_val_transform(), shuffle=False)
    ckpt, best = os.path.join(OUT, f"{tag}.pth"), -1.0
    epochs = a.epochs or EPOCHS_STAGE1
    for ep in range(epochs):
        model.train(); tot = 0.0; opt.zero_grad(set_to_none=True)
        for i, (img, lbl) in enumerate(tl):
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = crit(model(img.to(DEVICE)), lbl.to(DEVICE).long())
            scaler.scale(loss / ACCUMULATION_STEPS).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update_parameters(model)
            tot += float(loss)
        evm = ema.module if ema is not None else model
        P, yv = infer(evm, vl); auc = macro_auc(P, yv)
        print(f"Epoch {ep+1:02d}/{epochs} | Loss {tot/max(len(tl),1):.4f} | Val AUC {auc:.4f} Acc {(P.argmax(1)==yv).mean():.4f}")
        if auc > best:
            best = auc; torch.save(evm.state_dict(), ckpt)
            print(f"  -> checkpoint saved (val AUC {best:.4f})")
    final = ExpModel(head, pretrained=False).to(DEVICE)
    final.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=True)
    P, yv = infer(final, vl)
    f = os.path.join(OUT, f"{tag}_val_predictions.npz")
    np.savez(f, y_prob=P, y_true=yv, image_id=va["image_id"].values)
    print(f"[explore:{tag}] val acc={(P.argmax(1)==yv).mean():.4f} : {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
