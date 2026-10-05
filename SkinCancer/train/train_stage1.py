import os

import torch
import torch.optim as optim

from SkinCancer.configs.data_config import STAGE1_CLASS_NAMES
from SkinCancer.configs.model_config import BACKBONE_NAME
from SkinCancer.configs.training_config import DEVICE, EPOCHS_STAGE1, LEARNING_RATE, STAGE1_MODEL_PATH, WEIGHT_DECAY
from SkinCancer.eval.evaluate import validate
from SkinCancer.models.stage1 import create_stage1_classifier
from SkinCancer.train.train_utils import RobustFocalLoss, train_one_epoch
from SkinCancer.utils.logging import debug_model_summary


def train_or_load_stage1(train_loader, val_loader):
    criterion = RobustFocalLoss(gamma=2.0, alpha=[0.8, 0.2])
    model = create_stage1_classifier(backbone_name=BACKBONE_NAME, pretrained=True).to(DEVICE)
    debug_model_summary("Stage 1", model)

    if os.path.exists(STAGE1_MODEL_PATH):
        print(f"[Stage 1] Loading pretrained: {STAGE1_MODEL_PATH}")
        model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)
    else:
        print("[Stage 1] Training from scratch...")
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        best_auc = 0.0
        for ep in range(EPOCHS_STAGE1):
            loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                is_multiclass=False,
                epoch=ep + 1,
                total_epochs=EPOCHS_STAGE1,
                stage_name="Stage 1 Train",
            )
            auc_val = validate(
                model,
                val_loader,
                criterion,
                is_multiclass=False,
                epoch=ep + 1,
                total_epochs=EPOCHS_STAGE1,
                stage_name="Stage 1 Val",
                class_names=STAGE1_CLASS_NAMES,
            )
            print(f"Epoch {ep + 1:02d}/{EPOCHS_STAGE1} | Loss: {loss:.4f} | Val AUC: {auc_val:.4f}")
            if auc_val > best_auc:
                best_auc = auc_val
                torch.save(model.state_dict(), STAGE1_MODEL_PATH)
        model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)

    validate(
        model,
        val_loader,
        criterion,
        is_multiclass=False,
        epoch=EPOCHS_STAGE1,
        total_epochs=EPOCHS_STAGE1,
        stage_name="Stage 1 Final Val",
        class_names=STAGE1_CLASS_NAMES,
    )
    return model
