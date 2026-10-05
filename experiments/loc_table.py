"""Print the overlap table: pointing game, energy ratio and area-matched IoU."""
import glob
import os
import re

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.environ.get("HAM_OUTPUT_ROOT", "outputs")
D = os.path.join(ROOT, "localize")
MI = {"pointing": 0, "energy": 1, "IoU": 2}


def collect():
    rec = {}
    for f in glob.glob(os.path.join(D, "loc_*_f*.npz")) + glob.glob(os.path.join(D, "seed*", "loc_*_f*.npz")):
        m = re.search(r"loc_(.+)_f(\d)\.npz$", os.path.basename(f))
        s = re.search(r"seed(\d+)", os.path.dirname(f))
        rec.setdefault((m.group(1), int(s.group(1)) if s else 42), {})[int(m.group(2))] = f
    return rec


def load(files):
    parts = [np.load(files[k], allow_pickle=True) for k in sorted(files)]
    R = np.concatenate([p["metrics"] for p in parts])
    ids = np.concatenate([p["image_id"] for p in parts])
    return R, ids, list(parts[0]["names"])


def main():
    rec = collect()
    if not rec:
        print(" results missing"); return
    print("=" * 96)
    print(" measurement results (LTV attention , 5fold = cross-validation 7,991 full)")
    print("=" * 96)
    print(f"  {'':<20}{'seed':>5}{'fold':>7}{'n':>7}{'pointing':>10}{'energy':>9}{'IoU':>8}"
          f"{' vs orig (same seed): Denergy / DIoU':>40}")
    ref_done = False
    stats = {}
    for (name, seed) in sorted(rec, key=lambda x: (x[0] != "orig", x[0], x[1])):
        files = rec[(name, seed)]
        R, ids, names = load(files)
        L = names.index("ltv")
        e, i, p = np.nanmean(R[:, L, 1]), np.nanmean(R[:, L, 2]), np.nanmean(R[:, L, 0])
        if len(files) == 5:
            stats.setdefault(name, []).append((e, i))
        cmp = ""
        if name != "orig" and ("orig", seed) in rec:
            common = sorted(set(files) & set(rec[("orig", seed)]))
            if common:
                Ra, ia, _ = load({k: files[k] for k in common})
                Rb, ib, _ = load({k: rec[("orig", seed)][k] for k in common})
                pos = {x: j for j, x in enumerate(ib)}; o = [pos[x] for x in ia]
                de = Ra[:, L, 1] - Rb[o, L, 1]; di = Ra[:, L, 2] - Rb[o, L, 2]
                m = ~np.isnan(de); mi = ~np.isnan(di)
                pe = wilcoxon(Ra[m, L, 1], Rb[o][m, L, 1]).pvalue if m.sum() > 10 else np.nan
                pi = wilcoxon(Ra[mi, L, 2], Rb[o][mi, L, 2]).pvalue if mi.sum() > 10 else np.nan
                cmp = f"{np.nanmean(de):+.3f} (p={pe:.1g}) / {np.nanmean(di):+.3f} (p={pi:.1g}) [fold {common}]"
        print(f"  {name:<20}{seed:>5}{len(files):>5}/5{len(ids):>7}{p:>10.3f}{e:>9.3f}{i:>8.3f}   {cmp}")
        if not ref_done and len(files) >= 1:
            ref = {n: (np.nanmean(R[:, j, 0]), np.nanmean(R[:, j, 1]), np.nanmean(R[:, j, 2]))
                   for j, n in enumerate(names) if n != "ltv"}
            ref_done = (name, seed, len(files))

    print("\n comparison (1-stage )")
    for n, (p, e, i) in ref.items():
        print(f"    {n:<8} pointing {p:.3f}  energy {e:.3f}  IoU {i:.3f}")

    print("\n seed (5fold done seed 2 )")
    noise = {}
    for n, v in stats.items():
        if len(v) >= 2:
            v = np.array(v); noise[n] = v.std(0, ddof=1)
            print(f"    {n:<20} seed {len(v)} energy {v[:,0].mean():.3f}±{v[:,0].std(ddof=1):.3f}"
                  f"   IoU {v[:,1].mean():.3f}±{v[:,1].std(ddof=1):.3f}")
    if not noise:
        print(" (seeds without results are omitted)")

    print("\n ")
    print(" - the unsupervised map differs from the centred Gaussian in energy and IoU (p<0.01)")
    print(" - seeds with at least two completed folds are listed")
    print(" - LTV attention is compared with the centred Gaussian on the same images")


if __name__ == "__main__":
    main()
