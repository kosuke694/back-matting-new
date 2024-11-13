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
        data_dir: base directory containing image and mask subdirectories
        transform: optional image transformations
        target_size: target size for resizing images and masks (default is 256x256)
        min_iou: minimum IoU threshold for filtering annotations (default is 0.5)
        """
        self.data_dir = os.path.join(data_dir, dataset_type)
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(target_size)
        
        # クラスラベルのマッピング
        self.label_mapping = {
            "human": 0,
            "cello": 1,
            "piano": 2,
            "guitar": 3,
            "saxophone": 4
        }
        
        # アノテーションの読み込みとIoUフィルタリング
        annotations = []
        for file in annotation_files:
            df = pd.read_csv(file)
            label_name = os.path.splitext(os.path.basename(file))[0].split('_')[-1]
            df = df.assign(LabelName=label_name.lower())
            if 'PredictedIoU' in df.columns:
                df = df[df['PredictedIoU'] >= min_iou]  # IoUフィルタリング
            annotations.append(df)
        
        self.annotations = pd.concat(annotations, ignore_index=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 画像とマスクのファイル名とクラスラベルの取得
        img_name = self.annotations.iloc[idx]['imgID']
        mask_name = self.annotations.iloc[idx]['maskID']
        label = self.annotations.iloc[idx]['LabelName'].lower()
        
        # 画像とマスクのパス
        img_path = os.path.join(self.data_dir, 'input', label, img_name)
        mask_path = os.path.join(self.data_dir, 'masks', label, mask_name)
        
        # ファイルの存在チェック
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or mask file not found: {img_path}, {mask_path}")

        # 画像とマスクを読み込み
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # リサイズとテンソル変換
        image = self.resize(image)
        mask = self.resize(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)
        
        # グレースケールのマスクを3チャンネルに変換
        mask = self.to_tensor(mask)
        mask = mask.expand(3, -1, -1)  # 1チャンネルから3チャンネルに変換

        # バウンディングボックスと数値化したクラスラベルの取得
        bbox = torch.tensor([
            self.annotations.iloc[idx]['BoxXMin'], self.annotations.iloc[idx]['BoxYMin'],
            self.annotations.iloc[idx]['BoxXMax'], self.annotations.iloc[idx]['BoxYMax']
        ])

        # クラス名を数値ラベルに変換
        class_label = torch.tensor([self.label_mapping[label]])

        # モデルの入力に合わせて image, mask, mask, mask, bbox, class_label を返す
        return image, mask, mask, mask, bbox, class_label
