"""Assemble the result tables from the saved predictions."""
import numpy as np, os
from SkinCancer.eval.metrics import calculate_detailed_multiclass_metrics
from SkinCancer.eval.bootstrap import bootstrap_macro_ci
F = ["nv","mel","bkl","df","vasc","bcc","akiec"]
M = ["Accuracy","Sensitivity","Specificity","AUC","F1","PPV","NPV"]
R = "runs"
P = {
  "1-stage":      f"{R}/abl_onestage/onestage_7class_predictions.npz",
  "hard":         f"{R}/abl_baseline/abl_baseline_hard_predictions.npz",
  "baseline":     f"{R}/abl_baseline/abl_baseline_soft_predictions.npz",
  "+LTV":         f"{R}/abl_ltv/abl_ltv_soft_predictions.npz",
  "+TEME":        f"{R}/abl_teme/abl_teme_soft_predictions.npz",
  "+TEME+LTV":    f"{R}/abl_full/abl_full_soft_predictions.npz",
  "simple-head":  f"{R}/abl_simplehead/abl_simplehead_soft_predictions.npz",
}
res = {}
for k, f in P.items():
    d = np.load(f, allow_pickle=True)
    y, p = d["y_true"].astype(int), d["y_prob"]
    o, _ = calculate_detailed_multiclass_metrics(y, p, F)
    ci = bootstrap_macro_ci(y, p, F, n_boot=10000, seed=42).set_index("metric")
    res[k] = (o, ci)
    print(f" done: {k}", flush=True)

H = "".join(f"{m[:4]:>9}" for m in M)
def row(k):
    return "".join(f"{res[k][0][m]:>9.4f}" for m in M)

print("\n" + "="*80)
print(" Table 3 (Ours row); published baselines are quoted as reported")
print("="*80)
print(f"  {'':<14}{H}")
print(f"  {'Ours':<14}{row('+TEME+LTV')}")

print("\n" + "="*80)
print(" Table 7 - routing comparison")
print("="*80)
print(f"  {'':<14}{H}")
for k in ["1-stage","hard","baseline"]:
    print(f"  {k:<14}{row(k)}")

print("\n" + "="*80)
print(" Table 9 - TEME / LTV ablation (point estimate [95% CI])")
print("="*80)
for k in ["baseline","+LTV","+TEME","+TEME+LTV"]:
    o, ci = res[k]
    print(f"  {k}")
    for m in M:
        print(f"      {m:<12} {o[m]:.4f} [{ci.loc[m,'ci_low']:.4f}, {ci.loc[m,'ci_high']:.4f}]")

print("\n" + "="*80)
print(" Table 10 - simple Stage 2 baseline")
print("="*80)
print(f"  {'':<14}{H}")
for k in ["simple-head","+TEME+LTV"]:
    print(f"  {k:<14}{row(k)}")
