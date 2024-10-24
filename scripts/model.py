import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # エンコーダ部分（事前学習モデルに合わせた形状に修正）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),  # 事前学習モデルの形状: (64, 3, 7, 7)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 追加
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 事前学習モデルの形状: (256, 128, 3, 3)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # デコーダ部分（事前学習モデルに合わせた形状に修正）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # 形状を 256 -> 128 に変更
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),   # 形状を 128 -> 64 に変更
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=7, stride=1, padding=3)  # 最終層の形状を調整
        )

    def forward(self, x):
        # エンコーダの実行
        x = self.encoder1(x)
        x = self.encoder2(x)

        # デコーダの実行
        x = self.decoder(x)

        return x
