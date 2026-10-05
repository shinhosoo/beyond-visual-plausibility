import torch
import torch.nn as nn

from SkinCancer.configs.model_config import BACKBONE_NAME, D_ATTN, IMG_SIZE, N_HEADS, N_SLOTS, SLOT_AGG
from SkinCancer.models.backbone import create_backbone
from SkinCancer.models.ltv import LocalizeThenVerifyFusion
from SkinCancer.models.teme import MultiScaleMicroEncoder, smart_reshape
from SkinCancer.utils.logging import debug_feature_map, debug_tensor


class Stage2Classifier(nn.Module):
    def __init__(
        self,
        backbone_name=BACKBONE_NAME,
        num_classes=1,
        d_attn=D_ATTN,
        n_heads=N_HEADS,
        n_slots=N_SLOTS,
        slot_agg=SLOT_AGG,
    ):
        super().__init__()
        self.backbone = create_backbone(backbone_name=backbone_name, pretrained=True, num_classes=0)
        self.backbone.set_grad_checkpointing(True)
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            final_feat, feats = self.backbone.forward_intermediates(dummy, intermediates_only=False)
            s2_feat = feats[1]
            if isinstance(s2_feat, (list, tuple)):
                s2_feat = s2_feat[0]
            self.s2_dim = s2_feat.shape[1] if s2_feat.dim() == 4 else s2_feat.shape[-1]
            macro_feat = self.backbone.forward_head(final_feat, pre_logits=True)
            self.final_dim = macro_feat.shape[1]

        self.micro_proj = nn.Sequential(
            nn.Conv2d(self.s2_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            MultiScaleMicroEncoder(256, 256),
            nn.AdaptiveAvgPool2d(40),
        )
        self.ltv_fusion = LocalizeThenVerifyFusion(
            d_macro=self.final_dim,
            d_micro=256,
            d_attn=d_attn,
            n_heads=n_heads,
            n_slots=n_slots,
            dropout=0.1,
            slot_agg=slot_agg,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_attn, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _smart_reshape(self, x):
        return smart_reshape(x, self.s2_dim)

    def forward(self, x, return_saliency=False):
        debug_feature_map("Stage2Classifier.input", x)
        print("[Debug][Layer] backbone.forward_intermediates before")
        final_feat, stages = self.backbone.forward_intermediates(x, intermediates_only=False)
        debug_tensor("Stage2Classifier.final_feat.after_backbone", final_feat)
        debug_tensor("Stage2Classifier.stages.after_backbone", stages)

        print("[Debug][Layer] backbone.forward_head before")
        macro_feat = self.backbone.forward_head(final_feat, pre_logits=True)
        debug_tensor("Stage2Classifier.macro_feat.after_head", macro_feat)

        s2_feat = stages[1]
        if isinstance(s2_feat, (list, tuple)):
            s2_feat = s2_feat[0]
        debug_tensor("Stage2Classifier.s2_feat.raw", s2_feat)
        s2_feat = self._smart_reshape(s2_feat)
        debug_feature_map("Stage2Classifier.s2_feat.reshaped", s2_feat)

        print("[Debug][Layer] micro_proj before")
        micro_feat = self.micro_proj(s2_feat)
        debug_feature_map("Stage2Classifier.micro_feat.after_micro_proj", micro_feat)

        print("[Debug][Layer] ltv_fusion before")
        fused, saliency = self.ltv_fusion(macro_feat, micro_feat)
        debug_tensor("Stage2Classifier.fused.after_ltv", fused)
        debug_tensor("Stage2Classifier.saliency.after_ltv", saliency)

        print("[Debug][Layer] classifier before")
        logits = self.classifier(fused)
        debug_tensor("Stage2Classifier.logits.after_classifier", logits)
        if return_saliency:
            return logits, saliency
        return logits
