import sys

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
from tqdm import tqdm

from src.data.dataset import ROCODataset
from src.data.preprocessing import load_roco_data
from src.models.multimodal_model import MultimodalModel
from src.utils.config import load_config

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
def train_epoch(model, dataloader, optimizer, device, epoch):
    """
        1 epoch 학습

        Args:
            model: 모델
            dataloader: 데이터로더
            optimizer: optimizer
            device: device
            epoch: 현재 epoch

        Returns:
            float: 평균 loss
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')

    for batch in progress_bar:
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)


        optimizer.zero_grad()
        outputs = model(images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'train_loss' :f'{loss.item():.4f}'})
    epoch_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch} | Train Loss: {epoch_loss}')
    return epoch_loss


def eval_epoch(model, dataloader, device, epoch):
    """
    Validation

    Args:
        model: 모델
        dataloader: validation 데이터로더
        device: device

    Returns:
        float: 평균 validation loss
    """
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Evaluating')
    with torch.no_grad():
        for batch in progress_bar:
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            progress_bar.set_postfix({'valid_loss' :f'{loss.item()}'})

        epoch_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} | Validation Loss: {epoch_loss:.4f}')
    return epoch_loss

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
        Checkpoint 저장

        Args:
            model: 모델
            optimizer: optimizer
            epoch: 현재 epoch
            loss: 현재 loss
            save_path: 저장 경로
    """
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss},
               save_path)

def main(config):
    # device 설정
    device = torch.device('cuda' if config['hardware']['device'] == 'cuda' else 'cpu')

    # 데이터 불러오기
    train_data = load_roco_data(config['data']['raw_dir'],split='train')
    val_data = load_roco_data(config['data']['raw_dir'],split='validation')

    # 데이터 셋 생성
    train_dataset = ROCODataset(train_data,train=True,)
    val_dataset = ROCODataset(val_data,train=False,)

    # 토크나이저 생성 및 pad_token 설정
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['hardware']['num_workers'],
                              collate_fn=lambda x : collate_fn(x, tokenizer),
                              pin_memory=config['hardware']['pin_memory'])
    val_loader = DataLoader(val_dataset,
                            batch_size=config['validation']['batch_size'],
                            shuffle=False, num_workers=config['hardware']['num_workers'],
                            collate_fn=lambda x : collate_fn(x, tokenizer),
                            pin_memory=config['hardware']['pin_memory'])


    # 모델 설정
    model = MultimodalModel(vision_model=config['model']['vision_encoder']['name'],
                            language_model=config['model']['language_decoder']['name'],
                            pretrained=True,
                            image_feature_dim=config['model']['vision_encoder']['feature_dim'])
    model = model.to(device)
    print(f"\nUsing device: {device}")

    optimizer_name = config['training']['optimizer']['name']
    lr = float(config['training']['learning_rate'])

    # optimizer 설정
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')


    best_val_loss = float('inf')
    os.makedirs(config['logging']['save_dir'], exist_ok=True)
    loss_list = []
    patience = int(config['training']['early_stopping']['patience'])  # 5 epoch 동안 개선 없으면 중단
    patience_counter = 0
    for epoch in range(config['training']['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = eval_epoch(model, val_loader, device, epoch)

        if (epoch+1) % config['logging']['save_freq'] == 0:

            save_path = os.path.join(config['logging']['save_dir'],
                                     f"{config['experiment']['name']}_epoch{epoch+1}.pth"
                                     )
            save_checkpoint(model, optimizer, epoch, val_loss, save_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(
                config['logging']['save_dir'],f"{config['experiment']['name']}_best.pth"
            )
            torch.save(model.state_dict(), best_path)  # ← 모델만!
            print(f"✅ New best! Val Loss: {val_loss:.4f}\n")
        else:
            patience_counter += 1
            print(f"\nNo improvement for {patience_counter} epochs")

            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch}\n")
                break
if __name__ == '__main__':
    # print(torch.cuda.is_available())
    # 파라미터 설정 및 parser 설정
    os.chdir('../..')
    config_path = './configs/base_config.yaml'

    parser = argparse.ArgumentParser(description='Train a model with YAML config')
    parser.add_argument("--config",
                         type=str,
                         default=config_path,
                         help="config file path")

    parser.add_argument("--train_batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Override validation batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override training learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")

    args = parser.parse_args()

    config = load_config(config_path)

    if args.train_batch_size is not None:
        config["training"]["batch_size"] = args.train_batch_size
    if args.val_batch_size is not None:
        config["validation"]["batch_size"] = args.val_batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.device is not None:
        config["hardware"]["device"] = args.device

    main(config)