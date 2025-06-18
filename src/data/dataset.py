import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SiameseDataset(Dataset):
    """Dataset for one-shot segmentation training."""
    
    def __init__(self, support_dir, query_dir, mask_dir):
        self.support_dir = support_dir
        self.query_dir = query_dir
        self.mask_dir = mask_dir
        self.query_images = sorted([f for f in os.listdir(query_dir) if f.endswith('.jpg')])
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, idx):
        query_name = self.query_images[idx]
        support_name = query_name.replace(".jpg", "_ball.jpg")
        mask_name = query_name.replace(".jpg", "_mask.png")

        support_img = self.transform(Image.open(os.path.join(self.support_dir, support_name)).convert("RGB"))
        query_img = self.transform(Image.open(os.path.join(self.query_dir, query_name)).convert("RGB"))
        mask = self.transform(Image.open(os.path.join(self.mask_dir, mask_name)).convert("L"))

        input_tensor = torch.cat([query_img, support_img], dim=0)  # 6 x 256 x 256
        return input_tensor, mask 