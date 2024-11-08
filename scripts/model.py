import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResnetConditionHR(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, nf_part=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks1=7, n_blocks2=3, padding_type='reflect', use_attention=True):
        super(ResnetConditionHR, self).__init__()
        self.use_attention = use_attention
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        use_bias = True

        # Encoder for main input
        model_enc1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[0], ngf, kernel_size=7, padding=0, bias=use_bias),
                      norm_layer(ngf), nn.ReLU(True)]
        model_enc1 += [nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                       norm_layer(ngf * 2), nn.ReLU(True)]
        model_enc2 = [nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * 4), nn.ReLU(True)]

        # Encoder for background
        model_enc_back = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[1], ngf, kernel_size=7, padding=0, bias=use_bias),
                          norm_layer(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2**i
            model_enc_back += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                               norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # Encoder for segmentation
        model_enc_seg = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[2], ngf, kernel_size=7, padding=0, bias=use_bias),
                         norm_layer(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2**i
            model_enc_seg += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # Encoder for motion (multi-frame)
        model_enc_multi = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc[3], ngf, kernel_size=7, padding=0, bias=use_bias),
                           norm_layer(ngf), nn.ReLU(True)]
        for i in range(2):
            mult = 2**i
            model_enc_multi += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                norm_layer(ngf * mult * 2), nn.ReLU(True)]

        # Combine encoders
        self.model_enc1 = nn.Sequential(*model_enc1)
        self.model_enc2 = nn.Sequential(*model_enc2)
        self.model_enc_back = nn.Sequential(*model_enc_back)
        self.model_enc_seg = nn.Sequential(*model_enc_seg)
        self.model_enc_multi = nn.Sequential(*model_enc_multi)

        # Combining features from different encoders
        mult = 2**2  # downsampled twice
        self.comb_back = nn.Sequential(nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, bias=False),
                                       norm_layer(nf_part), nn.ReLU(True))
        self.comb_seg = nn.Sequential(nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, bias=False),
                                      norm_layer(nf_part), nn.ReLU(True))
        self.comb_multi = nn.Sequential(nn.Conv2d(ngf * mult * 2, nf_part, kernel_size=1, bias=False),
                                        norm_layer(nf_part), nn.ReLU(True))

        # Decoder
        self.model_res_dec = self.build_resnet_block(ngf * mult + 3 * nf_part, n_blocks1, padding_type, norm_layer, use_dropout, use_bias)
        self.model_res_dec_al = self.build_resnet_block(ngf * mult, n_blocks2, padding_type, norm_layer, use_dropout, use_bias)
        self.model_res_dec_fg = self.build_resnet_block(ngf * mult, n_blocks2, padding_type, norm_layer, use_dropout, use_bias)

        # Decoder output layers
        self.model_al_out = self.build_output_layer(ngf, 1)
        self.model_fg_out = self.build_output_layer(ngf, output_nc - 1)

    def build_resnet_block(self, in_dim, n_blocks, padding_type, norm_layer, use_dropout, use_bias):
        layers = [nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False), norm_layer(in_dim), nn.ReLU(True)]
        for _ in range(n_blocks):
            layers += [ResnetBlock(in_dim, padding_type, norm_layer, use_dropout, use_bias)]
        return nn.Sequential(*layers)

    def build_output_layer(self, in_dim, out_dim):
        layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(in_dim * 4, int(in_dim * 2), kernel_size=3, padding=1),
                  nn.Conv2d(int(in_dim * 2), out_dim, kernel_size=7, padding=0),
                  nn.Tanh()]
        return nn.Sequential(*layers)

    def forward(self, image, back, seg, multi):
        img_feat1 = self.model_enc1(image)
        img_feat = self.model_enc2(img_feat1)
        back_feat = self.model_enc_back(back)
        seg_feat = self.model_enc_seg(seg)
        multi_feat = self.model_enc_multi(multi)

        oth_feat = torch.cat([self.comb_back(torch.cat([img_feat, back_feat], dim=1)),
                              self.comb_seg(torch.cat([img_feat, seg_feat], dim=1)),
                              self.comb_multi(torch.cat([img_feat, back_feat], dim=1))], dim=1)

        out_dec = self.model_res_dec(torch.cat([img_feat, oth_feat], dim=1))
        out_dec_al = self.model_res_dec_al(out_dec)
        al_out = self.model_al_out(out_dec_al)
        out_dec_fg = self.model_res_dec_fg(out_dec)
        fg_out = self.model_fg_out(out_dec_fg)

        return al_out, fg_out
