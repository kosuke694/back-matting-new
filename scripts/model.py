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
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = LightSelfAttention(out_channels) if use_attention else None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.attention:
            x = self.attention(x)
        x += residual
        return self.relu(x)

class ResnetConditionHR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, nf_part=64, use_attention=False):
        super(ResnetConditionHR, self).__init__()

        # Encoder Blocks with residual connections
        self.model_enc1 = nn.Sequential(
            ResidualBlock(input_nc[0], ngf, use_attention),
            ResidualBlock(ngf, ngf * 2, use_attention)
        )
        self.model_enc_back = nn.Sequential(
            ResidualBlock(input_nc[1], ngf, use_attention),
            ResidualBlock(ngf, ngf * 2, use_attention)
        )
        # `model_enc_seg` now accepts 3-channel input as specified in `input_nc[2]`
        self.model_enc_seg = nn.Sequential(
            ResidualBlock(input_nc[2], ngf, use_attention),  # Set input_nc[2] to 3
            ResidualBlock(ngf, ngf * 2, use_attention)
        )
        self.model_enc_multi = nn.Sequential(
            ResidualBlock(input_nc[3], ngf, use_attention),
            ResidualBlock(ngf, ngf * 2, use_attention)
        )

        self.comb_back = nn.Conv2d(ngf * 2 * 2, nf_part, kernel_size=1)
        self.comb_seg = nn.Conv2d(ngf * 2 * 2, nf_part, kernel_size=1)
        self.comb_multi = nn.Conv2d(ngf * 2 * 2, nf_part, kernel_size=1)

        # Decoder components
        self.model_res_dec = nn.Sequential(
            nn.Conv2d(ngf * 2 + 3 * nf_part, ngf * 2, kernel_size=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Output heads for alpha and foreground
        self.model_al_out = nn.Sequential(
            nn.Conv2d(ngf, 1, kernel_size=1),
            nn.Tanh()
        )
        self.model_fg_out = nn.Sequential(
            nn.Conv2d(ngf, output_nc - 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, image, back, seg, multi):
        img_feat = self.model_enc1(image)
        back_feat = self.model_enc_back(back)
        seg_feat = self.model_enc_seg(seg)  # `seg` can now be 3 channels
        multi_feat = self.model_enc_multi(multi)

        combined_feat = torch.cat([img_feat, back_feat], dim=1)
        back_comb = self.comb_back(combined_feat)

        combined_feat = torch.cat([img_feat, seg_feat], dim=1)
        seg_comb = self.comb_seg(combined_feat)

        combined_feat = torch.cat([img_feat, multi_feat], dim=1)
        multi_comb = self.comb_multi(combined_feat)

        decoder_input = torch.cat([img_feat, back_comb, seg_comb, multi_comb], dim=1)
        out_dec = self.model_res_dec(decoder_input)

        al_out = self.model_al_out(out_dec)
        fg_out = self.model_fg_out(out_dec)

        return al_out, fg_out

# モデルエイリアスとして指定
MyDetectionModel = ResnetConditionHR
