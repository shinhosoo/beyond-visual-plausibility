"""Variants of the localisation experiment (no TEME; corrected masking without the global path)."""
import argparse
import torch
import torch.nn as nn
import experiments.localize as L
from SkinCancer.configs.model_config import IMG_SIZE, D_ATTN
from experiments.shared import Branch

_make = L.make_branch


class Zero(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d = d
    def forward(self, x):
        return x.new_zeros(x.shape[0], self.d)


def make_branch(net, mode):
    if mode == "bias_local":
        br, s2_dim = _make(net, "bias")
        br.ltv_fusion.macro_proj = Zero(D_ATTN).to(L.DEVICE)
        return br, s2_dim
    if mode != "noteme":
        return _make(net, mode)
    with torch.no_grad():
        f, s2, _ = L.feats(net, torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=L.DEVICE))
        s2_dim = s2.shape[1]; final_dim = net.forward_head(f, pre_logits=True).shape[1]
    return Branch(s2_dim, final_dim, 7, teme=False, ltv=True).to(L.DEVICE), s2_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "eval"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ltv-mode", default="bias_local", choices=["orig", "bias", "norm", "noteme", "bias_local"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    L.make_branch = make_branch
    (L.train if a.stage == "train" else L.evaluate)(a)


if __name__ == "__main__":
    main()
