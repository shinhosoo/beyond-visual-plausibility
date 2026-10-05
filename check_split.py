"""Verify that the splits contain no leakage.
"""

import argparse
import glob
import itertools
import os
import sys

import pandas as pd

FINAL7 = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
PAPER_TEST = {"nv": 1336, "mel": 236, "bkl": 243, "df": 18, "vasc": 34, "bcc": 90, "akiec": 67}
DERIVED = {
    "stage1": FINAL7,
    "stage2_mel": ["nv", "mel"],
    "stage2_nonmel": ["bkl", "df", "vasc", "bcc", "akiec"],
}


def load(split_root, prefix, subset):
    for name in (f"{prefix}_{subset}.csv",):
        p = os.path.join(split_root, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "diagnosis" not in df.columns and "dx" in df.columns:
                df = df.rename(columns={"dx": "diagnosis"})
            df["diagnosis"] = df["diagnosis"].str.lower()
            return df
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-root", default=os.environ.get("HAM_SPLIT_ROOT", "./splits"))
    ap.add_argument("--data-root", default=os.environ.get("HAM_DATA_ROOT"))
    args = ap.parse_args()

    ok = True
    print(f"split_root = {os.path.abspath(args.split_root)}\n")

    print("[1] Split files")
    g = {}
    for subset in ["train", "val", "test"]:
        d = load(args.split_root, "global", subset)
        if d is None:
            d = load(args.split_root, "stage1", subset)
        g[subset] = d
        if g[subset] is None:
            print(f"    {subset:<6} missing: neither global_{subset}.csv nor stage1_{subset}.csv found")
            ok = False
    missing = [f"{p}_{s}.csv" for p in DERIVED for s in ["train", "val", "test"]
               if load(args.split_root, p, s) is None]
    print(f" derived files: {'all present' if not missing else missing}")
    if any(v is None for v in g.values()):
        sys.exit(1)

    tot_i = sum(len(v) for v in g.values())
    tot_l = pd.concat(g.values()).lesion_id.nunique() if "lesion_id" in g["train"].columns else None
    print("\n[2] Sizes")
    print(f"    {'subset':<7}{'images':>8}{'%':>9}{'lesions':>9}")
    for s in ["train", "val", "test"]:
        L = g[s].lesion_id.nunique() if tot_l else 0
        print(f"    {s:<7}{len(g[s]):>8}{100*len(g[s])/tot_i:>8.2f}%{L:>9}")
    print(f"    {'':<7}{tot_i:>8}{'':>9}{tot_l if tot_l else '-':>9}")
    if tot_i != 10015:
        print(f" warning: full 10,015 ({tot_i})")
        ok = False

    print("\n[3] Overlap between splits")
    ids = pd.concat(g.values()).image_id
    dup = int(ids.duplicated().sum())
    print(f" image_id duplicate: {dup}{'' if dup == 0 else ' <-- '}")
    ok &= dup == 0
    if tot_l:
        for a, b in itertools.combinations(["train", "val", "test"], 2):
            n = len(set(g[a].lesion_id) & set(g[b].lesion_id))
            print(f" lesion_id overlap {a}/{b}: {n}{'' if n == 0 else ' <-- '}")
            ok &= n == 0
    else:
        print(" lesion_id overlap:")
        ok = False

    print("\n[4] Class counts in the test split")
    tc = g["test"].diagnosis.value_counts()
    print(f"    {'class':<7}{'CSV':>7}{'':>7}{'':>6}")
    hit = True
    for c in FINAL7:
        n = int(tc.get(c, 0))
        d = n - PAPER_TEST[c]
        hit &= d == 0
        print(f"    {c.upper():<7}{n:>7}{PAPER_TEST[c]:>7}{d:>+6}")
    print(f"    {'':<7}{len(g['test']):>7}{sum(PAPER_TEST.values()):>7}"
          f"{len(g['test'])-sum(PAPER_TEST.values()):>+6}")
    print(f"    => {' (split)' if hit else ' (split)'}")
    ok &= hit

    print("\n[5] Per-stage split files")
    for prefix, allowed in DERIVED.items():
        for s in ["train", "val", "test"]:
            d = load(args.split_root, prefix, s)
            if d is None:
                continue
            exp = int(g[s].diagnosis.isin(allowed).sum())
            same = set(d.image_id) <= set(g[s].image_id)
            good = len(d) == exp and same
            ok &= good
            print(f"    {prefix+'_'+s:<22}{len(d):>6} / expected {exp:<6}{'OK' if good else ''}")

    print("\n[6] Image files")
    if not args.data_root:
        print(" (pass --data-root to check the image files)")
    else:
        have, dirs = set(), []
        if os.path.isdir(args.data_root):
            entries = {n.lower(): n for n in os.listdir(args.data_root)}
            for part in ("ham10000_images_part_1", "ham10000_images_part_2"):
                if part in entries:
                    dirs.append(os.path.join(args.data_root, entries[part]))
        if not dirs:
            print(f" warning: {args.data_root} part_1 / part_2 ")
            ok = False
        for d in dirs:
            n = {os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(d, "*.jpg"))}
            print(f"    {os.path.basename(d)}: {len(n)}")
            have |= n
        print(f" : {len(have)}")
        for s in ["train", "val", "test"]:
            miss = len(set(g[s].image_id) - have)
            print(f"    {s:<6} images not found: {miss}")
            ok &= miss == 0

    print("\n" + ("=" * 46))
    print(" all checks passed" if ok else " leakage found")
    print("=" * 46)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
