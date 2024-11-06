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



class MyDetectionModel(nn.Module):
    def __init__(self, input_nc, output_nc, num_classes):
        super(MyDetectionModel, self).__init__()
        self.num_classes = num_classes  # クラス数の設定
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=2, padding=1),   # 出力: 64x128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),         # 出力: 128x64x64
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),        # 出力: 256x32x32
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),        # 出力: 512x16x16
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),        # 出力: 512x8x8
            nn.ReLU()
        )
        # バウンディングボックス出力用
        self.bbox_head = nn.Linear(512 * 8 * 8, 4)  
        # クラス分類出力用
        self.class_head = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)  # [batch_size, 512, 8, 8]
        
        # バウンディングボックスの予測
        x_flatten = x.view(x.size(0), -1)  # [batch_size, 512 * 8 * 8]
        bbox = self.bbox_head(x_flatten)   # [batch_size, 4]

        # クラス分類の予測
        class_preds = self.class_head(x)   # [batch_size, num_classes, height, width]

        return bbox, class_preds

