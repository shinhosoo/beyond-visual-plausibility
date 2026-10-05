import numpy as np
import torch

from SkinCancer.configs.data_config import FINAL_CLASS_NAMES
from SkinCancer.configs.model_config import BACKBONE_NAME
from SkinCancer.configs.training_config import (
    DEVICE,
    FINAL_MACRO_CI_PATH,
    FINAL_PREDS_PATH,
    STAGE1_MODEL_PATH,
    STAGE2_MELANOCYTIC_MODEL_PATH,
    STAGE2_NONMELANOCYTIC_MODEL_PATH,
)
from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map, load_final7_test_data
from SkinCancer.data.transforms import build_val_transform
from SkinCancer.eval.bootstrap import bootstrap_macro_ci, print_macro_ci
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics, print_detailed_metrics
from SkinCancer.models.routing import get_two_stage_predictions_7class
from SkinCancer.models.stage1 import create_stage1_classifier
from SkinCancer.models.stage2 import Stage2Classifier
from SkinCancer.utils.logging import debug_model_summary

def main():
    print("\n" + "=" * 70)
    print("   SOFT ROUTING 7-CLASS EVALUATION")
    print("=" * 70)

    path_map = build_image_path_map()
    df = load_final7_test_data(path_map)
    loader = make_loader(df, "final_label", build_val_transform())

    stage1_model = create_stage1_classifier(backbone_name=BACKBONE_NAME, pretrained=False).to(DEVICE)
    mel_model = Stage2Classifier(num_classes=1).to(DEVICE)
    nonmel_model = Stage2Classifier(num_classes=5).to(DEVICE)
    debug_model_summary("Stage 1", stage1_model)
    debug_model_summary("Stage 2 Mel", mel_model)
    debug_model_summary("Stage 2 Non-Mel", nonmel_model)

    stage1_model.load_state_dict(torch.load(STAGE1_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=True)
    mel_model.load_state_dict(torch.load(STAGE2_MELANOCYTIC_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=True)
    nonmel_model.load_state_dict(torch.load(STAGE2_NONMELANOCYTIC_MODEL_PATH, map_location=DEVICE, weights_only=True), strict=True)

    y_prob, y_true = get_two_stage_predictions_7class(stage1_model, mel_model, nonmel_model, loader)
    np.savez(FINAL_PREDS_PATH, y_prob=y_prob, y_true=y_true,
             y_pred=y_prob.argmax(1), image_id=df["image_id"].values)
    overall, per_class = calculate_detailed_multiclass_metrics(y_true, y_prob, FINAL_CLASS_NAMES)
    print_detailed_metrics(overall, per_class, "Soft Routing 7-Class Metrics")

    ci_df = bootstrap_macro_ci(y_true, y_prob, FINAL_CLASS_NAMES)
    ci_df.to_csv(FINAL_MACRO_CI_PATH, index=False)
    print_macro_ci(ci_df)

if __name__ == "__main__":
    main()
