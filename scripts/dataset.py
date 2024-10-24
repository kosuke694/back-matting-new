import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        カスタムデータセットの初期化
        Args:
            data_dir (str): 画像ファイルとラベルのあるディレクトリ
            transform (callable, optional): 画像に適用する変換
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # ディレクトリ内の画像ファイルを読み込む
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_dir)  # ラベルはディレクトリ名から推測

    def __len__(self):
        # データセットのサイズを返す
        return len(self.image_paths)

    def __getitem__(self, idx):
        # インデックスに対応する画像とラベルを取得
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 画像を読み込む
        image = Image.open(img_path).convert("RGB")

        # 必要に応じて変換を適用
        if self.transform:
            image = self.transform(image)

        # ラベルをテンソルに変換（例: intにキャストしてからテンソル化）
        label = torch.tensor(int(label))

        return image, label
