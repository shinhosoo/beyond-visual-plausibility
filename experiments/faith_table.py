"""Print the faithfulness table: insertion minus deletion for each saliency map."""
import glob
import os

import numpy as np
from scipy.stats import wilcoxon

from SkinCancer.configs.training_config import OUTPUT_ROOT

D = os.path.join(OUTPUT_ROOT, "faithful")


def main(args_seed=42):
    fs = sorted(glob.glob(os.path.join(D, "faith_f*_s*.npz")))
    if not fs:
        print("  no faithfulness results found"); return
    import re, collections
    by_seed = collections.defaultdict(list)
    for f in fs:
        m = re.search(r"_s(\d+)\.npz$", f)
        by_seed[int(m.group(1)) if m else 42].append(f)
    seeds = sorted(by_seed)
    if len(seeds) > 1:
        print(f"  results from seeds {seeds}; showing one seed at a time (use --seed to choose)\n")
    sel = args_seed if args_seed in by_seed else seeds[0]
    fs = sorted(by_seed[sel])
    parts = [np.load(f, allow_pickle=True) for f in fs]
    names = [k[:-4] for k in parts[0].files if k.endswith("_del")]
    names = [k for k in names if all(f"{k}_del" in p.files for p in parts)]
    cur = {k: {m: np.concatenate([p[f"{k}_{m}"] for p in parts])
               for m in ("del", "ins")} for k in names}
    n = len(next(iter(cur.values()))["del"])
    print(f"  [seed {sel}, {len(fs)} folds]")
    print("=" * 84)
    print(f" Deletion and insertion test   {len(fs)} folds   n={n}")
    print("=" * 84)
    print(f"  {'saliency map':<16}{'deletion↓':>12}{'insertion↑':>12}{'ins-del↑':>11}   vs gc4 / vs center (insertion, Wilcoxon)")
    base = {k: cur[k]["ins"].mean(1) for k in names}
    for k in names:
        d, i = cur[k]["del"].mean(1), cur[k]["ins"].mean(1)
        c = ""
        if k not in ("gc4", "center"):
            for t in ("gc4", "center"):
                if t in cur:
                    p = wilcoxon(i, base[t]).pvalue
                    c += f"  {np.mean(i-base[t]):+.4f} (p={p:.1g})"
        print(f"  {k:<16}{d.mean():>12.4f}{i.mean():>12.4f}{(i-d).mean():>11.4f}   {c}")
    print("\n  Lower deletion and higher insertion mean the map points at the evidence the model uses.")
    print("  A map that does not beat the centered Gaussian or a random map carries no such evidence.")


if __name__ == "__main__":
    import sys
    sd = 42
    if "--seed" in sys.argv:
        sd = int(sys.argv[sys.argv.index("--seed") + 1])
    main(sd)
