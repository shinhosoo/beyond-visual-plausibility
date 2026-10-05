"""Correction configurations with a larger learning rate for the branch heads."""
import argparse, types
import torch
import experiments.residual as R

VARIANTS = {
    "d1lr":  ("d1",    True,  1,   1,   True,      10.0),
    "d3lr":  ("d3",    True,  1,   1,   False,     10.0),
    "d1plr": ("d1",    True,  0,   0,   True,      10.0),
}

class ResidualModelLR(R.ResidualModel):
    last = None
    def __init__(self, name, teme=True, ltv=True, detach=True, pretrained=True):
        kind, det, tm, lt, self.s1_loss, self.mult = VARIANTS[name]
        super().__init__(kind, tm, lt, det, pretrained)
        ResidualModelLR.last = self
    def forward(self, x):
        out = super().forward(x)
        if not self.s1_loss:
            out.pop("s1", None)
        return out

def adamw_grouped(params, lr, weight_decay):
    m = ResidualModelLR.last
    bb = [p for n, p in m.named_parameters() if n.startswith("net.")]
    hd = [p for n, p in m.named_parameters() if not n.startswith("net.")]
    print(f"[residual_lr] lr backbone={lr:g} ({len(bb)} tensors)  heads={lr * m.mult:g} ({len(hd)} tensors)")
    return torch.optim.AdamW([{"params": bb, "lr": lr}, {"params": hd, "lr": lr * m.mult}], lr=lr, weight_decay=weight_decay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(VARIANTS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    kind, det, tm, lt, _, _ = VARIANTS[a.name]
    R.ResidualModel = ResidualModelLR
    R.optim = types.SimpleNamespace(AdamW=adamw_grouped)
    a.model, a.detach, a.teme, a.ltv = a.name, int(det), tm, lt
    R.run(a)

if __name__ == "__main__":
    main()
