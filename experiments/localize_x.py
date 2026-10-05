"""Helper layers for the localisation experiments."""
import argparse
import os

import torch
import torch.nn as nn

import SkinCancer.configs.model_config as MC
import experiments.localize as L
import experiments.shared as SH
from SkinCancer.configs.model_config import IMG_SIZE

V = {
    "orig": {}, "bias": dict(ltv="bias"), "norm": dict(ltv="norm"),
    "noteme": dict(teme=False),
    "bias_local": dict(ltv="bias", local=True),
    "noteme_bias_local": dict(ltv="bias", local=True, teme=False),
    "s3": dict(stage=2),
    "s3_bias_local": dict(ltv="bias", local=True, stage=2),
    "k1_bias_local": dict(ltv="bias", local=True, slots=1),
    "k2_bias_local": dict(ltv="bias", local=True, slots=2),
    "k8_bias_local": dict(ltv="bias", local=True, slots=8),
    "e20_bias_local": dict(ltv="bias", local=True, epochs=20),
}


class Zero(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d
    def forward(self, x):
        return x.new_zeros(x.shape[0], self.d)


def setup(name):
    cfg = dict(ltv="orig", local=False, teme=True, stage=1, slots=None, epochs=None)
    cfg.update(V[name])
    if cfg["slots"]:
        MC.N_SLOTS = cfg["slots"]; SH.N_SLOTS = cfg["slots"]
    if cfg["stage"] != 1:
        k = cfg["stage"]
        def feats(net, x, _k=k):
            f, st = net.forward_intermediates(x, intermediates_only=False)
            s = st[_k][0] if isinstance(st[_k], (list, tuple)) else st[_k]
            return f, s, st
        L.feats = feats

    def make_branch(net, _mode):
        with torch.no_grad():
            f, s, _ = L.feats(net, torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=L.DEVICE))
            s_dim = s.shape[1]; final_dim = net.forward_head(f, pre_logits=True).shape[1]
        br = SH.Branch(s_dim, final_dim, 7, teme=cfg["teme"], ltv=True)
        if cfg["ltv"] != "orig":
            from experiments.ltv_fix import LTVFixed
            br.ltv_fusion = LTVFixed(d_macro=final_dim, d_micro=256, d_attn=MC.D_ATTN, n_heads=MC.N_HEADS,
                                     n_slots=MC.N_SLOTS, dropout=0.1, slot_agg=MC.SLOT_AGG, mode=cfg["ltv"])
        if cfg["local"]:
            br.ltv_fusion.macro_proj = Zero(MC.D_ATTN)
        return br.to(L.DEVICE), s_dim
    L.make_branch = make_branch
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True, choices=list(V))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = setup(a.name)
    if a.seed != 42:
        L.OUT = os.path.join(L.OUT, f"seed{a.seed}"); os.makedirs(L.OUT, exist_ok=True)
    a.ltv_mode = a.name
    a.epochs = cfg["epochs"] or 10
    print(f"[localize_x] name={a.name} seed={a.seed} cfg={cfg} out={L.OUT}")
    (L.train if a.stage == "train" else L.evaluate)(a)


if __name__ == "__main__":
    main()
