import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from SkinCancer.configs.training_config import ACCUMULATION_STEPS, DEVICE
from SkinCancer.utils.logging import debug_tensor, get_current_lr


class RobustFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.85):
        super().__init__()
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([1 - alpha, alpha])
        else:
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        prob = torch.sigmoid(input)
        pt = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha.to(input.device)
        alpha_t = alpha_t[1] * target + alpha_t[0] * (1 - target)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    is_multiclass=False,
    epoch=1,
    total_epochs=1,
    stage_name="Train",
):
    model.train()
    loss_sum = 0
    optimizer.zero_grad()
    total_batches = len(loader)
    pbar = tqdm(loader, desc=f"Training {stage_name}", leave=False)
    for i, (img, lbl) in enumerate(pbar):
        print(
            f"\n[Debug][Train] {stage_name} | "
            f"epoch={epoch}/{total_epochs} | batch={i + 1}/{total_batches} | "
            f"lr={get_current_lr(optimizer):.8f}"
        )
        debug_tensor("train.batch.img.before_to_device", img)
        debug_tensor("train.batch.lbl.before_to_device", lbl)
        img = img.to(DEVICE)
        if is_multiclass:
            lbl = lbl.to(DEVICE).long()
        else:
            lbl = lbl.to(DEVICE).unsqueeze(1)
        debug_tensor("train.batch.img.after_to_device", img)
        debug_tensor("train.batch.lbl.after_to_device", lbl)
        with torch.amp.autocast("cuda"):
            out = model(img)
            debug_tensor("train.batch.model_output", out)
            loss = criterion(out, lbl) / ACCUMULATION_STEPS
        print(f"[Debug][Train] loss_before_accumulation={(loss.item() * ACCUMULATION_STEPS):.6f}")
        scaler.scale(loss).backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            print(f"[Debug][Train] optimizer_step=True at batch={i + 1}/{total_batches}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            print(f"[Debug][Train] optimizer_step=False accumulation={(i + 1) % ACCUMULATION_STEPS}/{ACCUMULATION_STEPS}")
        loss_sum += loss.item() * ACCUMULATION_STEPS
        pbar.set_postfix({"loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}"})
    return loss_sum / len(loader)
