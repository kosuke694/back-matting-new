import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(data_dir))  # 入力画像のリスト
        self.mask_names = sorted(os.listdir(mask_dir))  # マスク画像のリスト
        self.transform = transform
        self.to_tensor = transforms.ToTensor()  # PIL画像をテンソルに変換するためのトランスフォーム
        self.resize = transforms.Resize(target_size)  # 画像とマスクをリサイズ

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        # 画像とマスクを読み込む
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # マスクは1チャネル（グレースケール）

        # リサイズ
        image = self.resize(image)
        mask = self.resize(mask)

        # 画像とマスクをテンソルに変換
        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)
        
        mask = self.to_tensor(mask)  # マスクもテンソルに変換

        return image, mask
