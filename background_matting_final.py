from __future__ import print_function
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from functions import to_image, get_bbox, crop_images, uncrop, composite4
from scripts.model import MyDetectionModel  # MyDetectionModelをインポート

torch.set_num_threads(1)

# 引数の設定
parser = argparse.ArgumentParser(description='Background Matting using Final Model.')
parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the trained model (final_model.pth).')
parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory to save the output results.')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory containing input images.')
parser.add_argument('-b', '--back', type=str, required=True, help='Captured background image (for fixed camera mode).')

args = parser.parse_args()

# モデルのパス
model_path = args.model_path

# 入力と出力のディレクトリ
input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# 背景画像の読み込み
back_img = cv2.imread(args.back)
if back_img is None:
    print(f"Error: Unable to read background image: {args.back}")
    exit(1)
back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)

# ネットワークの初期化
print("Loading model...")
netM = MyDetectionModel(input_nc=(3, 3, 1, 4), output_nc=4)  # MyDetectionModelクラスを使用
netM = nn.DataParallel(netM)
print("Model loading started...")
netM.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
print("Model loaded successfully.")

netM.cuda()
netM.eval()
cudnn.benchmark = True
reso = (512, 512)  # ネットワーク入力解像度

# 入力画像の取得
test_imgs = [f for f in os.listdir(input_dir) if f.endswith('_img.jpg')]
test_imgs.sort()

# 各画像の処理
for i, filename in enumerate(test_imgs):
    print(f"Processing {filename}...")
    # 入力画像の読み込み
    bgr_img = cv2.imread(os.path.join(input_dir, filename))
    if bgr_img is None:
        print(f"Error: Unable to read input image: {filename}")
        continue
    bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # セグメンテーションマスクの読み込み
    mask_path = os.path.join(input_dir, filename.replace('_img', '_masksDL'))
    rcnn = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if rcnn is None:
        print(f"Warning: Segmentation mask not found for {filename}. Using a default mask.")
        rcnn = np.zeros((bgr_img.shape[0], bgr_img.shape[1]), dtype=np.uint8)

    # バイナリ化
    _, rcnn = cv2.threshold(rcnn, 128, 255, cv2.THRESH_BINARY)

    # 画像のクロップと前処理
    bbox = get_bbox(rcnn, R=bgr_img.shape[0], C=bgr_img.shape[1])
    crop_list = [bgr_img, back_img, rcnn, back_img]
    crop_list = crop_images(crop_list, reso, bbox)
    bgr_img, bg_im, rcnn, back_img_cropped = crop_list

    # Torchテンソルへ変換
    img = torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0).float().div(255).cuda()
    bg = torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0).float().div(255).cuda()
    rcnn_al = torch.from_numpy(rcnn).unsqueeze(0).unsqueeze(0).float().div(255).cuda()

    with torch.no_grad():
        # ネットワークでの推論
        try:
            alpha_pred, fg_pred_tmp = netM(img, bg, rcnn_al, multi=None)
            if alpha_pred is None or fg_pred_tmp is None:
                raise ValueError("Model inference returned None for alpha_pred or fg_pred_tmp.")
            print(f"Inference successful for {filename}.")
        except Exception as e:
            print(f"Error during inference for {filename}: {e}")
            continue

        # 前景とアルファマスクを復元
        try:
            alpha_out = to_image(alpha_pred[0])
            fg_out = to_image(fg_pred_tmp[0])

            # チャンネル数チェックと修正
            if len(fg_out.shape) == 2 or fg_out.shape[-1] != 3:
                print(f"Warning: Foreground shape mismatch: {fg_out.shape}, expected [H, W, 3]")
                fg_out = fg_out[..., :3] if fg_out.shape[-1] > 3 else np.repeat(fg_out[..., np.newaxis], 3, axis=-1)
        except Exception as e:
            print(f"Error during alpha and foreground extraction for {filename}: {e}")
            continue

        # サイズ復元処理のエラー対応
        try:
            alpha_out = uncrop(alpha_out, bbox, bgr_img.shape[0], bgr_img.shape[1])
            fg_out = uncrop(fg_out, bbox, bgr_img.shape[0], bgr_img.shape[1])
        except ValueError as e:
            print(f"Error while uncropping for {filename}: {e}")
            print("Resizing alpha and fg_out to match target dimensions.")
            alpha_out = cv2.resize(alpha_out, (bgr_img.shape[1], bgr_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            fg_out = cv2.resize(fg_out, (bgr_img.shape[1], bgr_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 出力の保存
    try:
        if len(alpha_out.shape) == 2:
            alpha_out = np.expand_dims(alpha_out, axis=-1)
        fg_out = cv2.resize(fg_out, (back_img.shape[1], back_img.shape[0]))
        alpha_out = cv2.resize(alpha_out, (back_img.shape[1], back_img.shape[0]))
        comp_img = composite4(fg_out, back_img, alpha_out)
        cv2.imwrite(os.path.join(output_dir, filename.replace('_img', '_out.png')), alpha_out)
        cv2.imwrite(os.path.join(output_dir, filename.replace('_img', '_fg.png')), fg_out)
        cv2.imwrite(os.path.join(output_dir, filename.replace('_img', '_comp.png')), comp_img)
    except ValueError as e:
        print(f"Error during composition: {e}")
        continue

    print(f"Processed {i+1}/{len(test_imgs)}: {filename}")

print("All images processed.")
