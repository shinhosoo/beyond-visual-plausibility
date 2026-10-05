import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = os.environ.get(
    "HAM_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "HAM10000_extracted")
)
METADATA_CSV = os.path.join(DATA_ROOT, "HAM10000_metadata.csv")

def _resolve_image_dirs(root):
    """Locate the HAM10000 image directories.

The dataset ships its images in two directories, part_1 and part_2; both are searched.
"""
    found = []
    if os.path.isdir(root):
        entries = {name.lower(): name for name in os.listdir(root)}
        for part in ("ham10000_images_part_1", "ham10000_images_part_2"):
            if part in entries:
                found.append(os.path.join(root, entries[part]))
    if not found:
        found = [
            os.path.join(root, "HAM10000_images_part_1"),
            os.path.join(root, "ham10000_images_part_2"),
        ]
    return found

IMAGE_DIRS = _resolve_image_dirs(DATA_ROOT)

SPLIT_ROOT = os.environ.get("HAM_SPLIT_ROOT", os.path.join(PROJECT_ROOT, "splits"))

STAGE1_TRAIN_CSV = os.path.join(SPLIT_ROOT, "stage1_train.csv")
STAGE1_VAL_CSV = os.path.join(SPLIT_ROOT, "stage1_val.csv")
STAGE1_TEST_CSV = os.path.join(SPLIT_ROOT, "stage1_test.csv")

STAGE2_MELANOCYTIC_TRAIN_CSV = os.path.join(SPLIT_ROOT, "stage2_mel_train.csv")
STAGE2_MELANOCYTIC_VAL_CSV = os.path.join(SPLIT_ROOT, "stage2_mel_val.csv")
STAGE2_MELANOCYTIC_TEST_CSV = os.path.join(SPLIT_ROOT, "stage2_mel_test.csv")

STAGE2_NONMELANOCYTIC_TRAIN_CSV = os.path.join(SPLIT_ROOT, "stage2_nonmel_train.csv")
STAGE2_NONMELANOCYTIC_VAL_CSV = os.path.join(SPLIT_ROOT, "stage2_nonmel_val.csv")
STAGE2_NONMELANOCYTIC_TEST_CSV = os.path.join(SPLIT_ROOT, "stage2_nonmel_test.csv")

FINAL7_TEST_CSV = os.path.join(SPLIT_ROOT, "global_test.csv")
if not os.path.exists(FINAL7_TEST_CSV):
    FINAL7_TEST_CSV = STAGE1_TEST_CSV

STAGE1_MAPPING = {
    "nv": 1,
    "mel": 1,
    "bkl": 0,
    "df": 0,
    "vasc": 0,
    "bcc": 0,
    "akiec": 0,
}
STAGE2_MELANOCYTIC_MAPPING = {"nv": 0, "mel": 1}
STAGE2_NONMELANOCYTIC_MAPPING = {"bkl": 0, "df": 1, "vasc": 2, "bcc": 3, "akiec": 4}
STAGE2_FINAL_MAPPING = {
    "nv": 0,
    "mel": 1,
    "bkl": 2,
    "df": 3,
    "vasc": 4,
    "bcc": 5,
    "akiec": 6,
}

FINAL_CLASS_NAMES = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
STAGE1_CLASS_NAMES = ["non_melanocytic", "melanocytic"]
MEL_CLASS_NAMES = ["nv", "mel"]
NM_CLASS_NAMES = ["bkl", "df", "vasc", "bcc", "akiec"]
