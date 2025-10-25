import sys
sys.path.append("../../")

import torch
import torch.nn as nn


from .vision_encoder import VisionEncoder
from .language_decoder import LanguageDecoder


class MultimodalModel(nn.Module):
    def __init__(self, vision_model='resnet50', language_model='gpt2', pretrained=True, image_feature_dim=2048):
        super(MultimodalModel, self).__init__()
        self.encoder = VisionEncoder(model_name=vision_model,
                                     pretrined=pretrained,
                                     feature_dim=image_feature_dim)

        self.decoder = LanguageDecoder(model_name=language_model,
                                       image_feature_dim=image_feature_dim)

    def forward(self, images, input_ids=None, attention_mask=None, labels=None):
        image_features = self.encoder(images,)


        return logits
