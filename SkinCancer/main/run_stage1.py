from SkinCancer.data.loaders import make_loader
from SkinCancer.data.split_io import build_image_path_map, load_stage1_data
from SkinCancer.data.transforms import build_train_transform, build_val_transform
from SkinCancer.train.train_stage1 import train_or_load_stage1
from SkinCancer.utils.logging import debug_cuda_memory
from SkinCancer.utils.seed import set_seed
from SkinCancer.configs.training_config import BOOTSTRAP_N, LEARNING_RATE, SEED


def main():
    print("\n" + "=" * 70)
    print("   STAGE 1: Melanocytic vs Non-melanocytic Classifier")
    print("=" * 70)
    print(f"[Debug][Config] LR={LEARNING_RATE}, SEED={SEED}, BOOTSTRAP_N={BOOTSTRAP_N}")
    debug_cuda_memory("startup")
    set_seed(SEED)
    path_map = build_image_path_map()
    df_train, df_val = load_stage1_data(path_map)
    train_loader = make_loader(df_train, "stage1_label", build_train_transform(), shuffle=True)
    val_loader = make_loader(df_val, "stage1_label", build_val_transform())
    train_or_load_stage1(train_loader, val_loader)


if __name__ == "__main__":
    main()
