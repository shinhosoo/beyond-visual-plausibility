"""Run an existing module with a different seed and output directory.

Replaces SEED, OUTPUT_ROOT and STAGE1_MODEL_PATH in the configuration before the target module
imports them, so that every seed retrains Stage 1 from scratch. The original sources are not
modified.

  python -m experiments.seed_run --seed 43 --out outputs_seed43 -- SkinCancer.main.run_stage1
"""
import argparse
import os
import runpy
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        sys.exit("run '--' ")

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.environ["HAM_OUTPUT_ROOT"] = out
    os.environ["HAM_SEED"] = str(a.seed)

    import SkinCancer.configs.training_config as TC
    TC.SEED = a.seed
    TC.OUTPUT_ROOT = out
    TC.STAGE1_MODEL_PATH = os.path.join(out, "stage1_mela_ltv_softrouting_multiscale_7class.pth")
    print(f"[seed_run] seed={TC.SEED}  output_root={TC.OUTPUT_ROOT}")
    print(f"[seed_run] stage1_path={TC.STAGE1_MODEL_PATH}")

    mod, argv = rest[0], rest[1:]
    sys.argv = [mod] + argv
    runpy.run_module(mod, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
