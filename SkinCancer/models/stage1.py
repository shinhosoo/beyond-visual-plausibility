from SkinCancer.models.backbone import create_stage1_backbone


def create_stage1_classifier(backbone_name="mambaout_tiny.in1k", pretrained=True):
    return create_stage1_backbone(backbone_name=backbone_name, pretrained=pretrained, num_classes=1)
