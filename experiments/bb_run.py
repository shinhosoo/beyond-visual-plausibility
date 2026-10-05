"""Run an existing module with a different backbone and output directory.

Replaces BACKBONE_NAME and OUTPUT_ROOT in the configuration before the target module imports
them. The original sources are not modified.

  python -m experiments.bb_run --backbone convnext_tiny --out outputs_convnext_tiny -- experiments.residual --model flat --fold 0
"""
import argparse
import os
import runpy
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True, help=" (output directory)")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        sys.exit("run '--' ")

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.environ["HAM_OUTPUT_ROOT"] = out

    import SkinCancer.configs.model_config as MC
    import SkinCancer.configs.training_config as TC
    MC.BACKBONE_NAME = a.backbone
    TC.OUTPUT_ROOT = out
    print(f"[bb_run] backbone={MC.BACKBONE_NAME}  output_root={TC.OUTPUT_ROOT}")

    mod, argv = rest[0], rest[1:]
    sys.argv = [mod] + argv
    runpy.run_module(mod, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
