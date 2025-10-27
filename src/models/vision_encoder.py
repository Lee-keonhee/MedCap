import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
# from transformers import Vi
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, feature_dim=2048):
        """
        Args:
            model_name: 'resnet50', 'resnet101', 'vit_base' 등
            pretrained: ImageNet pre-trained weights 사용 여부
            feature_dim: 출력 feature 차원
        """
        super(VisionEncoder, self).__init__()
        self.feature_dim = feature_dim

        # resnet 구현
        if model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)                # resnet-50의 feature_dim=2048임
            backbone_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        # TODO: vision-transformer 구현
        # elif model_name == 'vit_base' and pretrained:
        #     self.backbone = transformers.VisionTransformer.from_pretrained(model_name)

        if backbone_dim != self.feature_dim:
            self.projection = nn.Linear(backbone_dim, feature_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, images):
        """
        Args:
            images: (batch_size, 3, 224, 224)

        Returns:
            features: (batch_size, feature_dim)
        """
        encoded = self.backbone(images)
        encoded = self.projection(encoded)
        return encoded



if __name__ == '__main__':
    # 테스트
    encoder = VisionEncoder(model_name='resnet50', pretrained=True)
    # print(encoder.backbone)
    # 더미 입력
    dummy_images = torch.randn(4, 3, 224, 224)

    # Forward
    features = encoder(dummy_images)

    print(f"Input shape: {dummy_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: torch.Size([4, 2048])")

    # 파라미터 수 확인
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
