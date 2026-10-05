"""LTV with the masking order corrected; inherits the original module, which is unmodified.

Original: tokens * importance -> LayerNorm -> Verify, where the normalisation discards most of
the weighting.
  norm : LayerNorm(tokens) * w -> Verify
  bias : add log(w) to the Verify attention logits
Localize, slots, aggregation and fusion are unchanged.
"""
import torch
from SkinCancer.models.ltv import LocalizeThenVerifyFusion


class LTVFixed(LocalizeThenVerifyFusion):
    def __init__(self, *args, mode="bias", **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ("norm", "bias")
        self.mode = mode

    def forward(self, macro_feat, gcn_feat):
        B, C, H, W = gcn_feat.shape
        N, h, hd = H * W, self.n_heads, self.head_dim
        tokens = gcn_feat.flatten(2).transpose(1, 2)
        slots = self.slot_init(macro_feat).view(B, self.n_slots, self.d_attn)
        Q1 = self.p1_q_proj(self.norm1_q(slots)).view(B, self.n_slots, h, hd).transpose(1, 2)
        K1 = self.p1_k_proj(self.norm1_kv(tokens)).view(B, N, h, hd).transpose(1, 2)
        saliency = ((Q1 @ K1.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        imp = saliency.mean(dim=1).max(dim=1).values
        w = imp / imp.max(dim=1, keepdim=True).values.clamp_min(1e-12)
        kv2 = self.norm2_kv(tokens)
        if self.mode == "norm":
            kv2 = kv2 * w.unsqueeze(-1)
        Q2 = self.p2_q_proj(self.norm2_q(slots)).view(B, self.n_slots, h, hd).transpose(1, 2)
        K2 = self.p2_k_proj(kv2).view(B, N, h, hd).transpose(1, 2)
        V2 = self.p2_v_proj(kv2).view(B, N, h, hd).transpose(1, 2)
        logits = (Q2 @ K2.transpose(-2, -1)) * self.scale
        if self.mode == "bias":
            logits = logits + torch.log(w.clamp_min(1e-6)).to(logits.dtype)[:, None, None, :]
        a = self.dropout(logits.softmax(dim=-1))
        out = (a @ V2).transpose(1, 2).contiguous().view(B, self.n_slots, self.d_attn)
        slots = slots + self.p2_out(out)
        slots = slots + self.p2_ffn(slots)
        if self.slot_agg == "mean":
            agg = slots.mean(dim=1)
        elif self.slot_agg == "max":
            agg = slots.max(dim=1).values
        else:
            sw = torch.softmax(self.slot_weight_proj(macro_feat), dim=-1)
            agg = (sw.unsqueeze(-1) * slots).sum(dim=1)
        fused = self.norm_out(self.out_proj(agg) + self.macro_proj(macro_feat))
        return fused, saliency.detach()
