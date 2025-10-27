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
        self.hidden_size = self.gpt2.config.hidden_size         # 768

        # 3. Projection layer 생성 (image_feature_dim → gpt2_hidden_size)
        if image_feature_dim != self.hidden_size:
            self.projection = nn.Linear(image_feature_dim, self.hidden_size)
        else:
            self.projection = nn.Identity()

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
        # 1. Image embedding 생성
        projected_features = self.projection(image_features)  # (B, 768)
        projected_features = self.ln(projected_features)
        projected_features = self.dropout(projected_features)

        # 2.  Text embedding과 결합
        image_embeddings = projected_features.unsqueeze(1)      # B, 1, 768
        if input_ids is not None:
            text_embeddings = self.gpt2.transformer.wte(input_ids)     # B, seq_len, 768
            inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)  # (B, 1+seq_len, 768)
            # 3. Attention mask 확장
            if attention_mask is not None:
                # Image token은 항상 attend (1)
                image_mask = torch.ones((attention_mask.shape[0],1),
                                        dtype=attention_mask.dtype,
                                        device=attention_mask.device)
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)
            # 4. Labels 확장
            if labels is not None:
                # Image token 위치는 loss 계산 안 함 (-100)
                image_labels = torch.full(size=(labels.shape[0],1),
                                          fill_value=-100,
                                          dtype=labels.dtype,
                                          device=labels.device)
                labels = torch.cat([image_labels, labels], dim=1)
            # 3. GPT-2 forward

            outputs = self.gpt2(inputs_embeds = inputs_embeds,
                                attention_mask = attention_mask,
                                labels=labels,)
            return outputs

        # Inference 모드 (나중에)
        return projected_features


if __name__ == '__main__':
    decoder = LanguageDecoder(model_name='gpt2', image_feature_dim=2048)

    # 더미 입력
    image_features = torch.randn(4, 2048)
    input_ids = torch.randint(0, 50257, (4, 20))
    attention_mask = torch.ones(4, 20)
    labels = torch.randint(0, 50257, (4, 20))

    # Forward
    outputs = decoder(image_features, input_ids, attention_mask, labels)

    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Expected: torch.Size([4, 21, 50257])")  # 20+1
    print(f"Loss: {outputs.loss.item():.4f}")