"""Evaluate the hierarchical variants on the validation and test splits."""

import argparse
import os

import numpy as np
from scipy.stats import binomtest

from SkinCancer.configs.training_config import OUTPUT_ROOT
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics

OUT = os.path.join(OUTPUT_ROOT, "hier")
IMP = os.path.join(OUTPUT_ROOT, "experiments")
F = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
K = ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1", "PPV", "NPV"]


def fuse(p1, A, C, veto):
    A = np.asarray(A, float); C = np.asarray(C, float)
    a_o = A[:, 2] if A.shape[1] == 3 else np.zeros(len(A))
    c_o = C[:, 5] if C.shape[1] == 6 else np.zeros(len(C))
    a_in = A[:, :2] / np.clip(A[:, :2].sum(1, keepdims=True), 1e-12, None)
    c_in = C[:, :5] / np.clip(C[:, :5].sum(1, keepdims=True), 1e-12, None)
    G = p1 * (1 - a_o) + (1 - p1) * c_o if veto else p1
    out = np.zeros((len(p1), 7))
    out[:, 0:2] = G[:, None] * a_in
    out[:, 2:7] = (1 - G)[:, None] * c_in
    return out


def load_hier(tag, split):
    f = os.path.join(OUT, f"{tag}_{split}_predictions.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return d["p_stage1"], d["p_mel_branch"], d["p_nonmel_branch"], d["y_true"].astype(int), d["image_id"]


def load_flat(split):
    f = os.path.join(OUT, f"flat_{split}_predictions.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return d["y_prob"], d["y_true"].astype(int), d["image_id"]


def load_baseline(split):
    f = os.path.join(IMP, f"abl_baseline_{split}_predictions.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    pm = d["p_mel"]
    return fuse(d["p_stage1"], np.stack([1 - pm, pm], 1), d["p_nonmel"], False), d["y_true"].astype(int), d["image_id"]


def metrics(y, P):
    o, _ = calculate_detailed_multiclass_metrics(y, P, F)
    return o


def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y
    n10 = int((ca & ~cb).sum()); n01 = int((~ca & cb).sum())
    p = 1.0 if n10 + n01 == 0 else binomtest(n10, n10 + n01, 0.5).pvalue
    return n10, n01, p


def align(ids_ref, ids, arr):
    pos = {k: i for i, k in enumerate(ids)}
    return arr[[pos[k] for k in ids_ref]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()

    for split in ["val", "test"]:
        print("\n" + "=" * 92)
        print(f" {split.upper()}  " + ("( - )" if split == "val"
                                        else "(validation split)"))
        print("=" * 92)
        print(f"  {'settings':<20}" + "".join(f"{k[:4]:>9}" for k in K) + f"{'MELrec':>9}")

        fl = load_flat(split)
        if fl is None:
            print(" [flat prediction missing]"); continue
        Pf, y, ids = fl
        rows = {"1-stage (flat)": Pf}
        bl = load_baseline(split)
        if bl is not None:
            rows["2-stage "] = align(ids, bl[2], bl[0])
        for t in args.tags:
            h = load_hier(t, split)
            if h is None:
                print(f"  [{t}] prediction missing"); continue
            p1, A, C, yh, ih = h
            p1, A, C = align(ids, ih, p1), align(ids, ih, A), align(ids, ih, C)
            rows[f"{t} plain"] = fuse(p1, A, C, False)
            if A.shape[1] == 3:
                rows[f"{t} veto"] = fuse(p1, A, C, True)

        for n, P in rows.items():
            o = metrics(y, P); p = P.argmax(1)
            print(f"  {n:<20}" + "".join(f"{o[k]:>9.4f}" for k in K) + f"{100*(p[y==1]==1).mean():>8.1f}%")

        pf = Pf.argmax(1)
        print(f"\n McNemar vs 1-stage (n01 > n10 )")
        for n, P in rows.items():
            if n.startswith("1-stage"):
                continue
            n10, n01, p = mcnemar(y, pf, P.argmax(1))
            flag = " <- " if n01 > n10 and p < 0.05 else (" ( )" if n01 > n10 else "")
            print(f"    {n:<20} n10={n10:>4} n01={n01:>4}  p={p:.4g}{flag}")

        if split == "val":
            cand = {n: metrics(y, P)["Accuracy"] for n, P in rows.items()
                    if not n.startswith("1-stage") and not n.startswith("2-stage")}
            if cand:
                best = max(cand, key=cand.get)
                print(f"\n => val : {best}  (Acc {cand[best]:.4f}, 1-stage {metrics(y, Pf)['Accuracy']:.4f})")
                print(f" test settings results .")


if __name__ == "__main__":
    main()
