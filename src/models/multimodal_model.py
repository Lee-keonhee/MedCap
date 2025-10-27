import sys
sys.path.append("../../")

import torch
import torch.nn as nn

try:
    from .vision_encoder import VisionEncoder
    from .language_decoder import LanguageDecoder
except ImportError:
    from vision_encoder import VisionEncoder
    from language_decoder import LanguageDecoder


class MultimodalModel(nn.Module):
    def __init__(self, vision_model='resnet50', language_model='gpt2', pretrained=True, image_feature_dim=2048):
        super(MultimodalModel, self).__init__()
        self.encoder = VisionEncoder(model_name=vision_model,
                                     pretrained=pretrained,
                                     feature_dim=image_feature_dim)

        self.decoder = LanguageDecoder(model_name=language_model,
                                       image_feature_dim=image_feature_dim)

    def forward(self, images, input_ids=None, attention_mask=None, labels=None):
        image_features = self.encoder(images,)
        outputs = self.decoder(image_features, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return outputs


if __name__ == '__main__':
    model = MultimodalModel(vision_model='resnet50', language_model='gpt2', pretrained=True, image_feature_dim=2048)

    images = torch.randn(4,3,224,224)
    input_ids = torch.randint(0,50257, (4,20), dtype=torch.long)
    attention_mask = torch.ones([4,20], dtype=torch.long)
    labels = torch.randint(0, 50257, (4,20), dtype=torch.long)

    outputs = model(images, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss, logits = outputs.loss, outputs.logits
    print(logits.shape)
    print(loss.item())
    print(outputs)