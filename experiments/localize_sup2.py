"""Mask supervision that keeps the global path, used to separate the two effects."""
import argparse
import os

import experiments.localize as L
import experiments.localize_sup as S

_build = S.build


def build_global(net, mode="bias", local=False):
    return _build(net, mode=mode, local=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tag = f"supg{a.lam:g}_global"
    if a.seed != 42:
        L.OUT = os.path.join(L.OUT, f"seed{a.seed}"); os.makedirs(L.OUT, exist_ok=True)
    S.build = build_global
    if a.stage == "train":
        S.train(a, tag)
    else:
        L.make_branch = lambda net, _m: build_global(net)
        a.ltv_mode = tag
        L.evaluate(a)


if __name__ == "__main__":
    main()
