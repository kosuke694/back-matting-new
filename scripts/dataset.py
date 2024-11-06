import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomDetectionDataset(Dataset):
    def __init__(self, dataset_type, annotation_files, data_dir, transform=None, target_size=(256, 256), min_iou=0.5):
        """
        dataset_type: "train", "validation", "test" のいずれか
        annotation_files: list of annotation CSV file paths
        """
        self.data_dir = os.path.join(data_dir, dataset_type)  # dataset_type に基づきディレクトリを選択
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(target_size)
        
        # 各クラスに対応する数値ラベルの辞書
        self.label_mapping = {
            "human": 0,
            "cello": 1,
            "piano": 2,
            "guitar": 3,
            "saxophone": 4
        }
        
        # 複数のCSVファイルからデータを読み込み、PredictedIoU によるフィルタリング
        self.annotations = pd.concat(
            [pd.read_csv(file).assign(LabelName=os.path.splitext(os.path.basename(file))[0].split('_')[-1]) 
             for file in annotation_files],
            ignore_index=True
        )
        self.annotations = self.annotations[self.annotations['PredictedIoU'] >= min_iou]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 各サンプルのファイルパスとカテゴリを取得
        img_name = self.annotations.iloc[idx]['imgID']
        mask_name = self.annotations.iloc[idx]['maskID']
        label = self.annotations.iloc[idx]['LabelName']
        
        img_path = os.path.join(self.data_dir, 'input', label, img_name)
        mask_path = os.path.join(self.data_dir, 'masks', label, mask_name)

        # 画像とマスクを読み込む
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # リサイズとテンソル変換
        image = self.resize(image)
        mask = self.resize(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)
        
        mask = self.to_tensor(mask)

        # バウンディングボックスと数値化したクラスラベル
        bbox = torch.tensor([
            self.annotations.iloc[idx]['BoxXMin'], self.annotations.iloc[idx]['BoxYMin'],
            self.annotations.iloc[idx]['BoxXMax'], self.annotations.iloc[idx]['BoxYMax']
        ])
        
        # クラス名を数値ラベルに変換
        class_label = torch.tensor([self.label_mapping[label]])

        return image, mask, bbox, class_label
