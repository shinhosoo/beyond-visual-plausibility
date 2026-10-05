"""LTV as the classification head, with the masking order corrected."""
import experiments.explore as E
from SkinCancer.configs.model_config import D_ATTN, N_HEADS, N_SLOTS, SLOT_AGG
from experiments.ltv_fix import LTVFixed

MODES = {"ltvn": "norm", "ltvb": "bias"}


def parse(name):
    parts = name.split("_")
    assert parts[0] in MODES, name
    return parts[0], "la" in parts[1:], "ema" in parts[1:]


class ExpModelFix(E.ExpModel):
    def __init__(self, head, pretrained=True):
        super().__init__("ltv", pretrained)
        d_macro = self.br.ltv_fusion.macro_proj.in_features
        self.br.ltv_fusion = LTVFixed(d_macro=d_macro, d_micro=256, d_attn=D_ATTN, n_heads=N_HEADS,
                                      n_slots=N_SLOTS, dropout=0.1, slot_agg=SLOT_AGG, mode=MODES[head])


def main():
    E.parse = parse
    E.ExpModel = ExpModelFix
    E.main()


if __name__ == "__main__":
    main()
