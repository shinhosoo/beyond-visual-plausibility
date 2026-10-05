import numpy as np
import torch
from tqdm import tqdm

from SkinCancer.configs.training_config import DEVICE


def get_two_stage_predictions_7class(stage1_model, melanocytic_model, nonmelanocytic_model, loader):
    stage1_model.eval()
    melanocytic_model.eval()
    nonmelanocytic_model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for img, lbl in tqdm(loader, desc="Pred(2-stage 7-Class)", leave=False):
            img = img.to(DEVICE)
            batch_probs = torch.zeros(img.shape[0], 7, device=img.device, dtype=torch.float32)

            with torch.amp.autocast("cuda"):
                s1_probs = torch.sigmoid(stage1_model(img)).view(-1, 1).float()
                mel_logits = melanocytic_model(img)
                nm_logits = nonmelanocytic_model(img)

            p_mel = torch.sigmoid(mel_logits).view(-1, 1).float()
            mel_branch_probs = torch.cat([1.0 - p_mel, p_mel], dim=1)
            nonmel_branch_probs = torch.softmax(nm_logits, dim=1).float()

            batch_probs[:, 0:2] = s1_probs * mel_branch_probs
            batch_probs[:, 2:7] = (1.0 - s1_probs) * nonmel_branch_probs

            preds.extend(batch_probs.cpu().numpy())
            targets.extend(lbl.numpy().flatten())

    return np.array(preds), np.array(targets)
