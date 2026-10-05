import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from SkinCancer.configs.model_config import IMG_SIZE
from SkinCancer.configs.training_config import DEVICE
from SkinCancer.eval.metrics import (
    calculate_detailed_binary_metrics,
    calculate_detailed_multiclass_metrics,
    print_detailed_binary_metrics,
    print_detailed_metrics,
)
from SkinCancer.utils.logging import debug_tensor

def validate(
    model,
    loader,
    criterion,
    is_multiclass=False,
    epoch=1,
    total_epochs=1,
    stage_name="Val",
    class_names=None,
):
    model.eval()
    preds, targets = [], []
    total_batches = len(loader)
    with torch.no_grad():
        for i, (img, lbl) in enumerate(tqdm(loader, desc=f"Val {stage_name}", leave=False)):
            print(
                f"\n[Debug][Val] {stage_name} | "
                f"epoch={epoch}/{total_epochs} | batch={i + 1}/{total_batches}"
            )
            debug_tensor("val.batch.img.before_to_device", img)
            debug_tensor("val.batch.lbl.before_to_device", lbl)
            img = img.to(DEVICE)
            debug_tensor("val.batch.img.after_to_device", img)
            with torch.amp.autocast("cuda"):
                out = model(img)
            debug_tensor("val.batch.model_output", out)
            if is_multiclass:
                preds.extend(torch.softmax(out, dim=1).cpu().float().numpy())
            else:
                preds.extend(torch.sigmoid(out).cpu().float().numpy().flatten())
            targets.extend(lbl.numpy().flatten())

    preds = np.array(preds)
    targets = np.array(targets)
    if is_multiclass:
        if class_names is not None:
            overall, per_class = calculate_detailed_multiclass_metrics(targets, preds, class_names)
            print_detailed_metrics(overall, per_class, f"{stage_name} Detailed Metrics")
        p64 = np.asarray(preds, dtype=np.float64)
        p64 = p64 / p64.sum(axis=1, keepdims=True)
        try:
            return roc_auc_score(targets.astype(int), p64, multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[Warning] macro AUC failed -> accuracy : {e}")
            return accuracy_score(targets, np.argmax(preds, axis=1))

    if class_names is not None:
        y_pred = (preds >= 0.5).astype(int)
        metrics, per_class = calculate_detailed_binary_metrics(targets, y_pred, preds, class_names)
        print_detailed_binary_metrics(metrics, per_class, f"{stage_name} Detailed Metrics")
    try:
        return roc_auc_score(targets, preds)
    except Exception:
        return 0.5

def get_predictions(model, loader, is_multiclass=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for img, lbl in tqdm(loader, desc="Pred", leave=False):
            img = img.to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(img)
            if is_multiclass:
                preds.extend(torch.softmax(out, dim=1).cpu().float().numpy())
            else:
                preds.extend(torch.sigmoid(out).cpu().float().numpy().flatten())
            targets.extend(lbl.numpy().flatten())
    return np.array(preds), np.array(targets)

def visualize_saliency(model, image_tensor, save_path, n_slots=4):
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            _, saliency = model(image_tensor.unsqueeze(0).to(DEVICE), return_saliency=True)
    sal = saliency[0].mean(dim=0).cpu().float()
    N = sal.shape[-1]
    H_s = int(np.sqrt(N))
    W_s = N // H_s
    sal = sal.view(n_slots, H_s, W_s)

    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

    fig, axes = plt.subplots(1, n_slots + 1, figsize=(4 * (n_slots + 1), 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for k in range(n_slots):
        sal_k = sal[k].numpy()
        sal_k = (sal_k - sal_k.min()) / (sal_k.max() - sal_k.min() + 1e-8)
        sal_k = np.array(Image.fromarray((sal_k * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
        axes[k + 1].imshow(img_np)
        axes[k + 1].imshow(sal_k, cmap="hot", alpha=0.5)
        axes[k + 1].set_title(f"Slot {k + 1}")
        axes[k + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
