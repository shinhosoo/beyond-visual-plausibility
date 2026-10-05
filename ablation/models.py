"""Model definitions for the seed-42 ablation."""

import os

import torch
import torch.nn as nn

from SkinCancer.configs.model_config import D_ATTN
from SkinCancer.models.stage2 import Stage2Classifier
from SkinCancer.models.teme import LightweightTopoBlock

class AblationStage2Classifier(Stage2Classifier):
    """Stage 2 classifier with TEME, LTV and the simple head switchable independently.
"""

    def __init__(self, num_classes=1, use_teme=False, use_ltv=True,
                 simple_head=False, d_attn=D_ATTN, **kwargs):
        super().__init__(num_classes=num_classes, d_attn=d_attn, **kwargs)
        self.use_teme = bool(use_teme)
        self.use_ltv = bool(use_ltv) and not simple_head
        self.simple_head = bool(simple_head)

        if self.simple_head:
            self.micro_proj = None
            self.ltv_fusion = None
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.final_dim),
                nn.Linear(self.final_dim, d_attn),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(d_attn, num_classes),
            )
            return

        if self.use_teme:
            layers = list(self.micro_proj)
            layers.insert(len(layers) - 1, LightweightTopoBlock(256, reduced_dim=32))
            self.micro_proj = nn.Sequential(*layers)

        if not self.use_ltv:
            self.ltv_fusion = None
            self.macro_proj = nn.Linear(self.final_dim, d_attn)
            self.micro_pool = nn.AdaptiveAvgPool2d(1)
            self.micro_vec = nn.Linear(256, d_attn)
            self.norm_out = nn.LayerNorm(d_attn)

    def forward(self, x, return_saliency=False):
        if self.use_ltv and not self.simple_head:
            return super().forward(x, return_saliency=return_saliency)

        final_feat, stages = self.backbone.forward_intermediates(x, intermediates_only=False)
        macro_feat = self.backbone.forward_head(final_feat, pre_logits=True)

        if self.simple_head:
            logits = self.classifier(macro_feat)
            return (logits, None) if return_saliency else logits

        s2_feat = stages[1]
        if isinstance(s2_feat, (list, tuple)):
            s2_feat = s2_feat[0]
        micro_feat = self.micro_proj(self._smart_reshape(s2_feat))

        fused = self.macro_proj(macro_feat) + self.micro_vec(
            self.micro_pool(micro_feat).flatten(1)
        )
        fused = self.norm_out(fused)
        logits = self.classifier(fused)
        return (logits, None) if return_saliency else logits

def build(num_classes, teme, ltv, simple_head=False):
    return AblationStage2Classifier(
        num_classes=num_classes, use_teme=teme, use_ltv=ltv, simple_head=simple_head
    )

def describe(model):
    topo = model.micro_proj is not None and any(
        "Topo" in type(m).__name__ for m in model.micro_proj
    )
    return {
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "topo": topo,
        "ltv": model.ltv_fusion is not None,
        "simple": model.simple_head,
        "params": sum(p.numel() for p in model.parameters()),
    }
