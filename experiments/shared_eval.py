"""Evaluate the shared-backbone configurations across seeds."""
import argparse, os
import numpy as np
from scipy.stats import binomtest
from SkinCancer.configs.training_config import OUTPUT_ROOT
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics

F = ["nv", "mel", "bkl", "df", "vasc", "bcc", "akiec"]
K = ["Accuracy", "Sensitivity", "Specificity", "AUC", "F1", "PPV", "NPV"]
FLAT_TAG = {42: "flat", 43: "flat_s43", 44: "flat_s44"}


def align(ref, ids, a):
    pos = {k: i for i, k in enumerate(ids)}
    return a[[pos[k] for k in ref]]


def load_flat(seed, split):
    f = os.path.join(OUTPUT_ROOT, "hier", f"{FLAT_TAG[seed]}_{split}_predictions.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    return d["y_prob"], d["y_true"].astype(int), d["image_id"]


def load_shared(tag, seed, split):
    f = os.path.join(OUTPUT_ROOT, "shared", f"{tag}_s{seed}_{split}_predictions.npz")
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


def stats(y, P):
    o, _ = calculate_detailed_multiclass_metrics(y, P, F)
    p = P.argmax(1)
    return [o[k] for k in K] + [100 * (p[y == 1] == 1).mean(), 100 * (p[y == 0] == 1).mean()]


def mel_at_far(P, y, fars=(0.05, 0.08, 0.12)):
    mel, nv = y == 1, y == 0; pts = []
    for sh in np.linspace(-4, 4, 161):
        Q = P.copy(); Q[:, 1] *= np.exp(sh); p = Q.argmax(1)
        pts.append(((p[nv] == 1).mean(), (p[mel] == 1).mean()))
    pts = np.array(pts)
    return [pts[np.argmin(abs(pts[:, 0] - f)), 1] * 100 for f in fars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="A")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    a = ap.parse_args()
    H = K + ["MELrec", "NV->MEL"]
    for split in ["val", "test"]:
        print("\n" + "=" * 100); print(f" {split.upper()} = hier ( )"); print("=" * 100)
        print(f"  {'settings':<18}" + "".join(f"{h[:6]:>9}" for h in H))
        agg = {"1-stage": [], "A hier": [], "A flat": [], "A avg": []}; pairs = []
        for s in a.seeds:
            fl, sh = load_flat(s, split), load_shared(a.tag, s, split)
            if fl is None or sh is None:
                print(f"  [seed {s}] prediction missing (1-stage={fl is not None}, A={sh is not None})"); continue
            Pf, y, ids = fl
            out = {"1-stage": Pf}
            for k in ["hier", "flat", "avg"]:
                out[f"A {k}"] = align(ids, sh["image_id"], sh[k])
            for n, P in out.items():
                r = stats(y, P); agg[n].append(r)
                print(f"  {n+' s'+str(s):<18}" + "".join(f"{v:>9.4f}" for v in r[:7]) + f"{r[7]:>8.1f}%{r[8]:>8.1f}%")
            pairs.append((s, y, Pf, out)); print()
        if not agg["1-stage"]:
            continue
        print(f"  --- {len(agg['1-stage'])} seed ( s.d.) ---")
        for n, rs in agg.items():
            R = np.array(rs); mu = R.mean(0)
            sd = R.std(0, ddof=1) if len(R) > 1 else np.zeros_like(mu)
            print(f"  {n+' ':<18}" + "".join(f"{v:>9.4f}" for v in mu[:7]) + f"{mu[7]:>8.1f}%{mu[8]:>8.1f}%")
            print(f"  {'  ± sd':<18}" + "".join(f"{v:>9.4f}" for v in sd[:7]) + f"{sd[7]:>8.1f}%{sd[8]:>8.1f}%")
        if split == "test":
            fm = np.array(agg["1-stage"]); hm = np.array(agg["A hier"])
            print(f"\n A hier - 1-stage (): Acc {hm[:,0].mean()-fm[:,0].mean():+.4f}   "
                  f"F1 {hm[:,4].mean()-fm[:,4].mean():+.4f}   Sens {hm[:,1].mean()-fm[:,1].mean():+.4f}   "
                  f"MELrec {hm[:,7].mean()-fm[:,7].mean():+.1f}%p")
            print("\n McNemar ( seed, n01 > n10 A )")
            for s, y, Pf, out in pairs:
                pf = Pf.argmax(1); line = f"    s{s}"
                for n in ["A hier", "A flat", "A avg"]:
                    pa = out[n].argmax(1); ca, cb = pf == y, pa == y
                    n10 = int((ca & ~cb).sum()); n01 = int((~ca & cb).sum())
                    p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
                    line += f"   {n}: {n10:>3}/{n01:>3} p={p:.3g}"
                print(line)
            print("\n (NV->MEL) MEL recall")
            for s, y, Pf, out in pairs:
                a_ = mel_at_far(out["A hier"], y); b_ = mel_at_far(Pf, y)
                print(f"    s{s}  FAR 5/8/12%:  A hier " + "/".join(f"{v:.1f}" for v in a_)
                      + "   single-stage" + "/".join(f"{v:.1f}" for v in b_))


if __name__ == "__main__":
    main()
