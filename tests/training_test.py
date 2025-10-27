import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.data.dataset import ROCODataset
from src.data.preprocessing import load_roco_data
from src.models.multimodal_model import MultimodalModel


def check_overfitting(data_path):
    """
    작은 데이터셋으로 overfitting 테스트
    목표: Loss가 0에 가까워지는지 확인
    """

    # TODO: Step 1 - 디바이스 설정
    # - CUDA 사용 가능하면 'cuda', 아니면 'cpu'
    # - device 변수에 저장
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: Step 2 - 데이터 준비
    # - ROCO train 데이터 로드 (load_roco_data)
    # - 처음 10개만 사용 (작은 샘플로 테스트)
    data_list = load_roco_data(data_path, split='train')
    data_list = data_list[:10]

    # TODO: Step 3 - Tokenizer 준비
    # - GPT2Tokenizer 로드
    # - pad_token 설정 (eos_token과 동일하게)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token


    # TODO: Step 4 - Collate function 작성
    # - 배치 내 이미지들 스택
    # - 텍스트들 tokenization (padding=True, truncation=True, max_length=50)
    # - 반환: {'images': ..., 'input_ids': ..., 'attention_mask': ..., 'labels': ...}


    # TODO: Step 5 - Dataset & DataLoader
    # - ROCODataset 생성 (train=True)
    # - DataLoader 생성 (batch_size=2, collate_fn 사용)
    train_dataset = ROCODataset(data_list, train=True,transform=None)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))

    # TODO: Step 6 - 모델 생성
    # - MultimodalModel 생성
    # - device로 이동
    # - model.train() 모드
    multimodal_model = MultimodalModel(vision_model='resnet50', language_model='gpt2', pretrained=True, image_feature_dim=2048)
    multimodal_model.to(device)

    # TODO: Step 7 - Optimizer 설정
    # - Adam optimizer (lr=1e-4)
    optimizer = optim.Adam(multimodal_model.parameters(), lr=0.001)
    # TODO: Step 8 - Training Loop
    # - 100 iteration (작은 데이터를 여러 번 반복)
    # - 매 10 iteration마다 loss 출력
    # - Loss가 감소하는지 확인
    multimodal_model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in train_loader:
            images, input_ids, attention, labels = batch['images'], batch['input_ids'], batch['attention_mask'], batch['labels']
            # print(images.shape)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = multimodal_model(images=images,
                                       input_ids=input_ids,
                                       attention_mask=attention,
                                       labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(train_loader)
        if epoch % 10 ==0:
            print(f'Epoch {epoch} Loss: {epoch_loss}')





def collate_fn(batch, tokenizer):
    """
    배치 데이터 처리

    Args:
        batch: [(image, caption), (image, caption), ...]

    Returns:
        dict: {
            'images': tensor (B, 3, 224, 224),
            'input_ids': tensor (B, seq_len),
            'attention_mask': tensor (B, seq_len),
            'labels': tensor (B, seq_len)
        }
    """
    # TODO: images와 captions 분리
    images, captions = zip(*batch)

    # TODO: 이미지 스택
    images = torch.stack(images)

    # TODO: 텍스트 tokenization
    encoded = tokenizer(
        list(captions),
        padding=True,
        truncation=True,
        max_length=50,
        return_tensors='pt'
    )

    # TODO: labels 생성 (input_ids와 동일)
    labels = encoded['input_ids'].clone()

    # TODO: 반환
    return {
        'images': images,
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels
    }


if __name__ == '__main__':
    data_path = './data/raw/ROCO/all_data'
    print("🚀 Overfitting Test 시작!")
    check_overfitting(data_path)
    print("✅ 테스트 완료!")