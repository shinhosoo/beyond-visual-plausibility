import timm

from SkinCancer.configs.model_config import BACKBONE_NAME


def create_backbone(backbone_name=BACKBONE_NAME, pretrained=True, num_classes=0):
    return timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)


def create_stage1_backbone(backbone_name=BACKBONE_NAME, pretrained=True, num_classes=1):
    return timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
