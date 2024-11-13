import torch
import torch.nn as nn
import torch.nn.functional as F

class LightSelfAttention(nn.Module):
    def __init__(self, in_dim, head_count=2, reduction=4):
        super(LightSelfAttention, self).__init__()
        self.head_count = head_count
        self.query_conv = nn.Conv2d(in_dim, in_dim // reduction, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // reduction, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x_down = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
        proj_query = self.query_conv(x_down).view(batch_size, -1, width * height // 4).permute(0, 2, 1)
        proj_key = self.key_conv(x_down).view(batch_size, -1, width * height // 4)
        energy = torch.bmm(proj_query, proj_key) / (channels ** 0.5)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x_down).view(batch_size, -1, width * height // 4)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width // 2, height // 2)
        out = F.interpolate(out, size=(width, height), mode="bilinear", align_corners=True)
        return out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv1x1:
            residual = self.conv1x1(residual)
        out += residual
        return F.relu(out)

class MyDetectionModel(nn.Module):
    def __init__(self, input_nc, output_nc, num_classes=5):
        super(MyDetectionModel, self).__init__()
        # Self-Attentionレイヤーを定義
        self.attention1 = LightSelfAttention(64)
        self.attention2 = LightSelfAttention(128)

        # エンコーダ
        self.model_enc1 = nn.Sequential(
            ResidualBlock(input_nc[0], 64),
            nn.MaxPool2d(2),
            self.attention1  # LightSelfAttentionを追加
        )
        self.model_enc2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            self.attention2  # LightSelfAttentionを追加
        )
        self.model_enc3 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2)
        )
        self.model_enc4 = nn.Sequential(
            ResidualBlock(256, 512),
            nn.MaxPool2d(2)
        )

        # デコーダ
        self.model_dec1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.model_dec2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.model_dec3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bbox_output = nn.Conv2d(64, output_nc, kernel_size=1)
        self.class_output = nn.Conv2d(64, num_classes, kernel_size=1)  # num_classesはクラス数

    def forward(self, image, back, seg, multi):
        # エンコーダ
        img_feat = self.model_enc1(image)
        img_feat = self.model_enc2(img_feat)
        img_feat = self.model_enc3(img_feat)
        img_feat = self.model_enc4(img_feat)

        # デコーダ
        x = self.model_dec1(img_feat)
        x = self.model_dec2(x)
        x = self.model_dec3(x)

        # bbox_preds と class_preds の出力
        bbox_preds = self.bbox_output(x)
        class_preds = self.class_output(x)

        return bbox_preds, class_preds
