import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer

class LanguageDecoder(nn.Module):
    def __init__(self, model_name='gpt2', image_feature_dim=2048):
        """
        Args:
            model_name: 'gpt2', 'gpt2-medium', 'gpt2-large'
            image_feature_dim: Vision encoder 출력 차원 (2048)
        """
        super(LanguageDecoder, self).__init__()

        # TODO: 구현해야 할 것들
        # 1. GPT-2 모델 로드
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        # 2. GPT-2의 hidden size 확인
        self.hidden_size = self.gpt2.config.hidden_size
        # 3. Projection layer 생성 (image_feature_dim → gpt2_hidden_size)
        self.projection = nn.Linear(image_feature_dim, self.hidden_size)

        self.ln = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, image_features, input_ids=None, attention_mask=None, labels=None):
        """
        Args:
            image_features: (batch_size, image_feature_dim) - Vision encoder 출력
            input_ids: (batch_size, seq_len) - Tokenized text (학습 시)
            labels: (batch_size, seq_len) - Target tokens (학습 시)

        Returns:
            outputs: GPT-2 출력 (logits, loss 등)
        """
        # TODO: 구현해야 할 것들
        # 1. Image features를 GPT-2 hidden size로 변환
        projected_features = self.projection(image_features)  # (B, 768)

        # 2. (나중에) Text embeddings과 결합
        image_embeddings = projected_features.unsqueeze(1)      # B, 1, 768
        text_embeddings = self.gpt2.transformer.wte(projected_features)     # B, seq_len, 768
        inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)  # (B, 1+seq_len, 768)

        # 3. GPT-2 forward

        if input_ids is not None:
            outputs = self.gpt2(input_ids = input_ids,
                                attention_mask = attention_mask,
                                labels=labels,)
            return outputs

        # Inference 모드 (나중에)
        return projected_features


if __name__ == '__main__':
    decoder = LanguageDecoder(model_name='gpt2', image_feature_dim=2048)
    # 더미 입력
    dummy_image_features = torch.randn(4, 2048)
    dummy_input_ids = torch.randint(0, 50257, (4, 20))  # GPT-2 vocab size

    # Forward
    outputs = decoder(dummy_image_features, input_ids=dummy_input_ids)

    print(f"Image features shape: {dummy_image_features.shape}")
    print(f"Input IDs shape: {dummy_input_ids.shape}")
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Expected: torch.Size([4, 20, 50257])")