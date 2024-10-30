import torch
import torch.nn as nn

class MyModel(nn.Module):  # ResnetConditionHR に基づく更新
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                 n_blocks1=7, n_blocks2=3, padding_type='reflect'):
        super(MyModel, self).__init__()

        # Encoder の設定
        use_bias = True

        # エンコーダ部分
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        # デコーダ部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=7, padding=3, bias=use_bias),
            nn.Sigmoid()  # 最終出力に Sigmoid を仮定
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        # 必要に応じて追加の Residual ブロックを経由
        x = self.decoder(x)
        return x
