import os

_gpu = os.environ.get("HAM_GPU")
if _gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
elif "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

from SkinCancer.configs.data_config import PROJECT_ROOT

OUTPUT_ROOT = os.environ.get("HAM_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "outputs"))
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2_MELANOCYTIC = 20
EPOCHS_STAGE2_NONMELANOCYTIC = 20
LEARNING_RATE = 5e-5
WEIGHT_DECAY = float(os.environ.get("HAM_WEIGHT_DECAY", 5e-5))
SEED = 42

STAGE1_MODEL_PATH = os.path.join(OUTPUT_ROOT, "stage1_mela_ltv_softrouting_multiscale_7class.pth")
STAGE2_MELANOCYTIC_MODEL_PATH = os.path.join(OUTPUT_ROOT, "stage2_mela_ltv_softrouting_multiscale_7class.pth")
STAGE2_NONMELANOCYTIC_MODEL_PATH = os.path.join(OUTPUT_ROOT, "stage2_nonmela_ltv_softrouting_multiscale_7class.pth")
FINAL_PREDS_PATH = os.path.join(OUTPUT_ROOT, "soft_routing_multiscale_7class_predictions.npz")
FINAL_MACRO_CI_PATH = os.path.join(OUTPUT_ROOT, "soft_routing_multiscale_7class_macro_95ci.csv")
BOOTSTRAP_N = 10000
