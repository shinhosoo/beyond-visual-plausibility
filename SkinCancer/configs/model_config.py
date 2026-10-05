import os

BACKBONE_NAME = "mambaout_tiny.in1k"
IMG_SIZE = 384
D_ATTN = 256
N_HEADS = 8
N_SLOTS = int(os.environ.get("HAM_N_SLOTS", 4))
SLOT_AGG = "weighted"
