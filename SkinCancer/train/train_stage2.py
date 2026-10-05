import os

import torch
import torch.nn as nn
import torch.optim as optim

from SkinCancer.configs.data_config import MEL_CLASS_NAMES, NM_CLASS_NAMES
from SkinCancer.configs.training_config import (
    DEVICE,
    EPOCHS_STAGE2_MELANOCYTIC,
    EPOCHS_STAGE2_NONMELANOCYTIC,
    LEARNING_RATE,
    STAGE2_MELANOCYTIC_MODEL_PATH,
    STAGE2_NONMELANOCYTIC_MODEL_PATH,
    WEIGHT_DECAY,
)
from SkinCancer.eval.evaluate import validate
from SkinCancer.models.stage2 import Stage2Classifier
from SkinCancer.train.train_utils import train_one_epoch
from SkinCancer.utils.logging import debug_class_weights, debug_model_summary


def train_or_load_stage2_melanocytic(df_train, train_loader, val_loader):
    n_neg = (df_train["stage2_label"] == 0).sum()
    n_pos = (df_train["stage2_label"] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
    debug_class_weights("Stage 2 Mel pos_weight", pos_weight, class_names=["mel"])

    model = Stage2Classifier(num_classes=1).to(DEVICE)
    debug_model_summary("Stage 2 Mel", model)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if os.path.exists(STAGE2_MELANOCYTIC_MODEL_PATH):
        print(f"[Stage 2 Mel] Loading pretrained: {STAGE2_MELANOCYTIC_MODEL_PATH}")
        model.load_state_dict(torch.load(STAGE2_MELANOCYTIC_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)
    else:
        print("[Stage 2 Mel] Training from scratch...")
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        best_auc = 0.0
        for ep in range(EPOCHS_STAGE2_MELANOCYTIC):
            loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                is_multiclass=False,
                epoch=ep + 1,
                total_epochs=EPOCHS_STAGE2_MELANOCYTIC,
                stage_name="Stage 2 Mel Train",
            )
            auc_val = validate(
                model,
                val_loader,
                criterion,
                is_multiclass=False,
                epoch=ep + 1,
                total_epochs=EPOCHS_STAGE2_MELANOCYTIC,
                stage_name="Stage 2 Mel Val",
                class_names=MEL_CLASS_NAMES,
            )
            print(f"Epoch {ep + 1:02d}/{EPOCHS_STAGE2_MELANOCYTIC} | Loss: {loss:.4f} | Val AUC: {auc_val:.4f}")
            if auc_val > best_auc:
                best_auc = auc_val
                torch.save(model.state_dict(), STAGE2_MELANOCYTIC_MODEL_PATH)
        if not os.path.exists(STAGE2_MELANOCYTIC_MODEL_PATH):
            torch.save(model.state_dict(), STAGE2_MELANOCYTIC_MODEL_PATH)
        model.load_state_dict(torch.load(STAGE2_MELANOCYTIC_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)

    validate(
        model,
        val_loader,
        criterion,
        is_multiclass=False,
        epoch=EPOCHS_STAGE2_MELANOCYTIC,
        total_epochs=EPOCHS_STAGE2_MELANOCYTIC,
        stage_name="Stage 2 Mel Final Val",
        class_names=MEL_CLASS_NAMES,
    )
    return model


def train_or_load_stage2_nonmelanocytic(df_train, train_loader, val_loader):
    class_counts = df_train["stage2_label"].value_counts().sort_index().values
    total_samples = class_counts.sum()
    class_weights = total_samples / (len(class_counts) * class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    debug_class_weights("Stage 2 Non-Mel class_weights", weight_tensor, class_names=NM_CLASS_NAMES)

    model = Stage2Classifier(num_classes=5).to(DEVICE)
    debug_model_summary("Stage 2 Non-Mel", model)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    if os.path.exists(STAGE2_NONMELANOCYTIC_MODEL_PATH):
        print(f"[Stage 2 Non-Mel] Loading pretrained: {STAGE2_NONMELANOCYTIC_MODEL_PATH}")
        model.load_state_dict(torch.load(STAGE2_NONMELANOCYTIC_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)
    else:
        print("[Stage 2 Non-Mel] Training from scratch...")
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scaler = torch.amp.GradScaler("cuda")
        best_auc = 0.0
        for ep in range(EPOCHS_STAGE2_NONMELANOCYTIC):
            loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                is_multiclass=True,
                epoch=ep + 1,
                total_epochs=EPOCHS_STAGE2_NONMELANOCYTIC,
                stage_name="Stage 2 Non-Mel Train",
            )
            auc_val = validate(
                model,
                val_loader,
                criterion,
                is_multiclass=True,
                epoch=ep + 1,
                total_epochs=EPOCHS_STAGE2_NONMELANOCYTIC,
                stage_name="Stage 2 Non-Mel Val",
                class_names=NM_CLASS_NAMES,
            )
            print(f"Epoch {ep + 1:02d}/{EPOCHS_STAGE2_NONMELANOCYTIC} | Loss: {loss:.4f} | Val Macro-AUC: {auc_val:.4f}")
            if auc_val > best_auc:
                best_auc = auc_val
                torch.save(model.state_dict(), STAGE2_NONMELANOCYTIC_MODEL_PATH)
        if not os.path.exists(STAGE2_NONMELANOCYTIC_MODEL_PATH):
            torch.save(model.state_dict(), STAGE2_NONMELANOCYTIC_MODEL_PATH)
        model.load_state_dict(torch.load(STAGE2_NONMELANOCYTIC_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)

    validate(
        model,
        val_loader,
        criterion,
        is_multiclass=True,
        epoch=EPOCHS_STAGE2_NONMELANOCYTIC,
        total_epochs=EPOCHS_STAGE2_NONMELANOCYTIC,
        stage_name="Stage 2 Non-Mel Final Val",
        class_names=NM_CLASS_NAMES,
    )
    return model
