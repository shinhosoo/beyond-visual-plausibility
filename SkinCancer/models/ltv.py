import torch
import torch.nn as nn

from SkinCancer.utils.logging import debug_feature_map, debug_tensor


class LocalizeThenVerifyFusion(nn.Module):
    def __init__(self, d_macro, d_micro, d_attn=256, n_heads=8, n_slots=2, dropout=0.1, slot_agg="weighted"):
        super().__init__()
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.n_slots = n_slots
        self.head_dim = d_attn // n_heads
        self.scale = self.head_dim ** -0.5
        self.slot_agg = slot_agg
        self.slot_init = nn.Sequential(
            nn.Linear(d_macro, d_attn * 2),
            nn.GELU(),
            nn.Linear(d_attn * 2, n_slots * d_attn),
        )
        self.norm1_q = nn.LayerNorm(d_attn)
        self.norm1_kv = nn.LayerNorm(d_micro)
        self.p1_q_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.p1_k_proj = nn.Linear(d_micro, d_attn, bias=False)
        self.norm2_q = nn.LayerNorm(d_attn)
        self.norm2_kv = nn.LayerNorm(d_micro)
        self.p2_q_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.p2_k_proj = nn.Linear(d_micro, d_attn, bias=False)
        self.p2_v_proj = nn.Linear(d_micro, d_attn, bias=False)
        self.p2_out = nn.Linear(d_attn, d_attn, bias=False)
        self.p2_ffn = nn.Sequential(
            nn.LayerNorm(d_attn),
            nn.Linear(d_attn, d_attn * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_attn * 2, d_attn),
        )
        if slot_agg == "weighted":
            self.slot_weight_proj = nn.Linear(d_macro, n_slots)
        self.out_proj = nn.Linear(d_attn, d_attn)
        self.macro_proj = nn.Linear(d_macro, d_attn)
        self.norm_out = nn.LayerNorm(d_attn)
        self.dropout = nn.Dropout(dropout)

    def _multihead_attn_p2(self, Q, K, V):
        debug_tensor("LTV.p2_attn.Q.before", Q)
        debug_tensor("LTV.p2_attn.K.before", K)
        debug_tensor("LTV.p2_attn.V.before", V)
        B, Lq, _ = Q.shape
        h, hd = self.n_heads, self.head_dim
        Q = Q.view(B, Lq, h, hd).transpose(1, 2)
        K = K.view(B, -1, h, hd).transpose(1, 2)
        V = V.view(B, -1, h, hd).transpose(1, 2)
        attn_w = (Q @ K.transpose(-2, -1)) * self.scale
        attn_w = self.dropout(attn_w.softmax(dim=-1))
        out = (attn_w @ V).transpose(1, 2).contiguous().view(B, Lq, self.d_attn)
        debug_tensor("LTV.p2_attn.attn_w.after", attn_w)
        debug_tensor("LTV.p2_attn.out.after", out)
        return out

    def forward(self, macro_feat, gcn_feat):
        debug_tensor("LTV.macro_feat.input", macro_feat)
        debug_feature_map("LTV.gcn_feat.input", gcn_feat)
        B, C, H, W = gcn_feat.shape
        N = H * W
        spatial_tokens = gcn_feat.flatten(2).transpose(1, 2)
        debug_tensor("LTV.spatial_tokens.after_flatten", spatial_tokens)
        slots = self.slot_init(macro_feat).view(B, self.n_slots, self.d_attn)
        debug_tensor("LTV.slots.after_slot_init", slots)
        Q1 = self.p1_q_proj(self.norm1_q(slots))
        debug_tensor("LTV.Q1.after_proj", Q1)
        kv1 = self.norm1_kv(spatial_tokens)
        K1 = self.p1_k_proj(kv1)
        debug_tensor("LTV.K1.after_proj", K1)
        h, hd = self.n_heads, self.head_dim
        Q1_h = Q1.view(B, self.n_slots, h, hd).transpose(1, 2)
        K1_h = K1.view(B, N, h, hd).transpose(1, 2)
        debug_tensor("LTV.Q1_h.after_reshape", Q1_h)
        debug_tensor("LTV.K1_h.after_reshape", K1_h)
        raw_attn = (Q1_h @ K1_h.transpose(-2, -1)) * self.scale
        saliency = raw_attn.softmax(dim=-1)
        debug_tensor("LTV.saliency.after_softmax", saliency)
        sal_avg = saliency.mean(dim=1)
        spatial_importance = sal_avg.max(dim=1).values
        masked_spatial = spatial_tokens * spatial_importance.unsqueeze(-1)
        debug_tensor("LTV.masked_spatial.after_mask", masked_spatial)
        Q2 = self.p2_q_proj(self.norm2_q(slots))
        kv2 = self.norm2_kv(masked_spatial)
        K2 = self.p2_k_proj(kv2)
        V2 = self.p2_v_proj(kv2)
        attn2_out = self._multihead_attn_p2(Q2, K2, V2)
        slots = slots + self.p2_out(attn2_out)
        debug_tensor("LTV.slots.after_p2_out", slots)
        slots = slots + self.p2_ffn(slots)
        debug_tensor("LTV.slots.after_ffn", slots)
        if self.slot_agg == "mean":
            aggregated = slots.mean(dim=1)
        elif self.slot_agg == "max":
            aggregated = slots.max(dim=1).values
        else:
            slot_w = torch.softmax(self.slot_weight_proj(macro_feat), dim=-1)
            aggregated = (slot_w.unsqueeze(-1) * slots).sum(dim=1)
            debug_tensor("LTV.slot_weights.after_softmax", slot_w)
        debug_tensor("LTV.aggregated.after_slot_agg", aggregated)
        fused = self.out_proj(aggregated)
        fused = self.norm_out(fused + self.macro_proj(macro_feat))
        debug_tensor("LTV.fused.output", fused)
        return fused, saliency.detach()
