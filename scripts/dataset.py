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

            # LabelName の正規化
            if 'LabelName' in df.columns:
                df['LabelName'] = df['LabelName'].str.strip().str.lower()
            else:
                print(f"Warning: 'LabelName' column not found in {file}. Skipping file.")
                continue

            # IoU フィルタリング
            if dataset_type == "train" and 'PredictedIoU' in df.columns:
                print(f"Before IoU filtering: {len(df)} rows in {file}")
                df = df[df['PredictedIoU'] >= min_iou]
                print(f"After IoU filtering: {len(df)} rows in {file}")
            else:
                print(f"IoU filtering skipped for dataset type: {dataset_type}")

            annotations.append(df)

        # アノテーションを統合
        if annotations:
            self.annotations = pd.concat(annotations, ignore_index=True)
        else:
            self.annotations = pd.DataFrame()  # 空の DataFrame
            print("Warning: No valid annotations loaded.")

        print(f"Loaded {len(self.annotations)} annotations after processing.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.annotations.empty:
            raise IndexError("Dataset is empty. No items to fetch.")

        # 画像とマスクのファイル名とクラスラベルの取得
        img_name = self.annotations.iloc[idx]['imgID']
        mask_name = self.annotations.iloc[idx]['maskID']
        label = self.annotations.iloc[idx]['LabelName']

        # 正しいフォルダ名に修正 (human body -> human など)
        if label == "human body":
            label = "human"

        # 画像とマスクのパス
        img_path = os.path.join(self.data_dir, 'input', label, img_name)
        mask_path = os.path.join(self.data_dir, 'masks', label, mask_name)

        # ファイルの存在チェック
        if not os.path.exists(img_path):
            print(f"Warning: Missing image: {img_path}. Skipping this entry.")
            return None
        if not os.path.exists(mask_path):
            print(f"Warning: Missing mask: {mask_path}. Skipping this entry.")
            return None

        # 画像とマスクを読み込み
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading image {img_path}: {e}. Skipping this entry.")
            return None

        try:
            mask = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"Error reading mask {mask_path}: {e}. Skipping this entry.")
            return None

        # リサイズとテンソル変換
        image = self.resize(image)
        mask = self.resize(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

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
