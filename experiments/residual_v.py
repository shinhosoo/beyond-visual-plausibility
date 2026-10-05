"""Further correction variants: detached branches and alternative gating.
"""
import argparse
import experiments.residual as R

VARIANTS = {
    "d1e": ("d1",      False,  1,    1,   False),
    "d1p": ("d1",      True,   0,    0,   True),
}

class ResidualModelV(R.ResidualModel):
    def __init__(self, name, teme=True, ltv=True, detach=True, pretrained=True):
        kind, det, tm, lt, self.s1_loss = VARIANTS[name]
        super().__init__(kind, tm, lt, det, pretrained)

    def forward(self, x):
        out = super().forward(x)
        if not self.s1_loss:
            out.pop("s1", None)
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(VARIANTS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()
    kind, det, tm, lt, _ = VARIANTS[a.name]
    R.ResidualModel = ResidualModelV
    a.model, a.detach, a.teme, a.ltv = a.name, int(det), tm, lt
    R.run(a)

if __name__ == "__main__":
    main()
