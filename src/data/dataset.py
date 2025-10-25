import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .preprocessing import load_roco_data, preprocess_image, preprocess_text

class ROCODataset(Dataset):
    def __init__(self, data_list, train=False, transform=None):
        self.data_list = data_list
        # Transform 설정
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        image_path, caption = data['image_path'], data['caption']
        image = preprocess_image(image_path)
        caption = preprocess_text(caption)

        image = self.transform(image)
        return image, caption
