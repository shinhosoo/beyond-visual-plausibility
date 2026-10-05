"""Print a short status line for the running batches."""
import glob, os, re, subprocess
import numpy as np
from scipy.stats import binomtest

ROOT = os.environ.get("HAM_OUTPUT_ROOT", "outputs")
NOISE = 0.006
QUEUE = {"ltv": "ltv_la_ema", "d3lr": "d1plr"}


def where(module):
    return "explore" if module.startswith("explore") else "residual"


def running():
    out = {}
    for line in subprocess.run(["ps", "-eo", "pid,user,etimes,cmd"], capture_output=True, text=True).stdout.splitlines():
        m = re.search(r"experiments\.(\w+) --(?:model|name) (\w+) --fold (\d)", line)
        if m:
            k = (m.group(1), m.group(2), int(m.group(3)))
            out[k] = max(out.get(k, 0), int(line.split()[2]))
    return out


def done_folds(sub, name):
    return sorted(int(re.search(r"_f(\d)_", f).group(1))
                  for f in glob.glob(os.path.join(ROOT, sub, f"{name}_f*_s42_val_predictions.npz")))


def load(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True); o = np.argsort(d["image_id"])
    return d["y_prob"][o], d["y_true"][o].astype(int)


def main():
    print("=" * 80); print(" GPU"); print("=" * 80)
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout
        for l in g.strip().splitlines():
            print("  " + l)
    except FileNotFoundError:
        print(" nvidia-smi missing")

    print("\n" + "=" * 80); print(" run "); print("=" * 80)
    run = running()
    if not run:
        print(" missing")
    for (mod, name, fold), secs in sorted(run.items(), key=lambda x: (x[0][1], x[0][2])):
        log = os.path.join("logs", where(mod), f"{name}_f{fold}_s42.log")
        txt = open(log).read() if os.path.exists(log) else ""
        ep = re.findall(r"Epoch (\d+)/20", txt); ep = int(ep[-1]) if ep else 0
        if ep == 0:
            print(f"  {name:<11} fold {fold} / ({secs/60:.0f} elapsed)"); continue
        per = secs / ep; rest = per * (20 - ep) / 60 + 1
        left = [k for k in range(5 if where(mod) == "residual" else 3) if k > fold and k not in done_folds(where(mod), name)]
        more = (per * 20 / 60 + 1) * len(left)
        q = f" -> {QUEUE[name]}" if name in QUEUE else ""
        acc = re.findall(r"Acc ([\d.]+)", txt)
        print(f"  {name:<11} fold {fold} {ep:>2}/20  val Acc {acc[-1] if acc else '-':<7}"
              f" fold {rest:.0f}, {(rest+more)/60:.1f}{q}")

    print("\n" + "=" * 80); print(" done fold"); print("=" * 80)
    names = {}
    for sub in ("residual", "explore"):
        for f in glob.glob(os.path.join(ROOT, sub, "*_f*_s42_val_predictions.npz")):
            names.setdefault((sub, os.path.basename(f).split("_f")[0]), None)
    for sub, n in sorted(names, key=lambda x: x[1]):
        d = done_folds(sub, n)
        print(f"  {n:<11} {len(d)}/{5 if sub == 'residual' else 3}  {d}")

    print("\n" + "=" * 80); print(f" flat ( fold, ±{NOISE*100:.1f}%p)"); print("=" * 80)
    base = lambda k: load(os.path.join(ROOT, "residual", f"flat_f{k}_s42_val_predictions.npz"))
    for sub, n in sorted(names, key=lambda x: x[1]):
        if n == "flat":
            continue
        rows, Y, A, B = [], [], [], []
        for k in range(5):
            b, m = base(k), load(os.path.join(ROOT, sub, f"{n}_f{k}_s42_val_predictions.npz"))
            if b is None or m is None:
                continue
            (pb, y), (pm, _) = b, m
            rows.append((k, (pm.argmax(1) == y).mean() - (pb.argmax(1) == y).mean()))
            Y.append(y); A.append(pb.argmax(1)); B.append(pm.argmax(1))
        if not rows:
            continue
        y, a, b_ = np.concatenate(Y), np.concatenate(A), np.concatenate(B)
        n10 = int(((a == y) & (b_ != y)).sum()); n01 = int(((a != y) & (b_ == y)).sum())
        p = binomtest(n10, n10 + n01, .5).pvalue if n10 + n01 else 1.0
        per = "  ".join(f"f{k}:{da*100:+.1f}{'+' if da > NOISE else ('-' if da < -NOISE else '')}" for k, da in rows)
        print(f"  {n:<11} ΔAcc(%p) {per:<40} flat {n10} / {n} {n01}  p={p:.3g}")


if __name__ == "__main__":
    main()
