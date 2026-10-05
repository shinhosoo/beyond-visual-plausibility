"""Build the lesion-level splits.

One global split is made at the lesion level; the per-stage splits are derived from it, so no
evaluation image is seen at any stage of the pipeline.

  python make_splits.py --metadata <HAM10000_metadata.csv> --out splits_paper_seed42 --seed 42
"""
import argparse, os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MEL = ["nv", "mel"]
NM  = ["bkl", "df", "vasc", "bcc", "akiec"]
S1  = {"nv":1,"mel":1,"bkl":0,"df":0,"vasc":0,"bcc":0,"akiec":0}
M2  = {"nv":0,"mel":1}
N2  = {"bkl":0,"df":1,"vasc":2,"bcc":3,"akiec":4}

ap = argparse.ArgumentParser()
ap.add_argument("--metadata", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

df = pd.read_csv(a.metadata).rename(columns={"dx": "diagnosis"})
df["diagnosis"] = df["diagnosis"].str.lower()
keep = [c for c in ["lesion_id","image_id","diagnosis","dx_type","age","sex","localization"] if c in df.columns]
df = df[keep]
print(f"total images={len(df)}, lesions={df.lesion_id.nunique()}")

def gsplit(d, size, seed):
    g = GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    i, j = next(g.split(d, groups=d.lesion_id.values))
    return d.iloc[i].reset_index(drop=True), d.iloc[j].reset_index(drop=True)

tv, te = gsplit(df, 0.2, a.seed)
tr, va = gsplit(tv, 0.1/(1-0.2), a.seed)
parts = {"train": tr, "val": va, "test": te}

for x, y in [("train","val"),("train","test"),("val","test")]:
    ov = set(parts[x].lesion_id) & set(parts[y].lesion_id)
    assert not ov, f"lesion leakage {x}/{y}: {len(ov)}"
print("lesion-level leakage check passed")

for n, d in parts.items():
    d = d.sort_values(["lesion_id","image_id"]).reset_index(drop=True)
    d.to_csv(f"{a.out}/global_{n}.csv", index=False)
    s = d.copy(); s["stage1_label"] = s.diagnosis.map(S1); s.to_csv(f"{a.out}/stage1_{n}.csv", index=False)
    m = d[d.diagnosis.isin(MEL)].copy(); m["stage2_label"] = m.diagnosis.map(M2); m.to_csv(f"{a.out}/stage2_mel_{n}.csv", index=False)
    k = d[d.diagnosis.isin(NM)].copy();  k["stage2_label"] = k.diagnosis.map(N2);  k.to_csv(f"{a.out}/stage2_nonmel_{n}.csv", index=False)
    print(f"{n:5s} {len(d):5d} images  {d.lesion_id.nunique():5d} lesions")

P = {"nv":1336,"mel":236,"bkl":243,"df":18,"vasc":34,"bcc":90,"akiec":67}
c = parts["test"].diagnosis.value_counts()
got = {k: int(c.get(k,0)) for k in P}
print("test support:", got)
print("paper Table 4:", "MATCH" if got == P else "MISMATCH")
