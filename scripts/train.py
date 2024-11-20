import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from dataset import CustomDetectionDataset
from model import MyDetectionModel
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 設定ファイルの読み込み
with open("C:/Users/klab/Desktop/back-matting-new/configs/train_config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの初期化
model = MyDetectionModel(input_nc=[3, 3, 3, 3], output_nc=1).to(device)

# 損失関数と最適化手法
criterion_bbox = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# ラベルマッピングの辞書
label_mapping = {
    "human": 0,
    "cello": 1,
    "piano": 2,
    "guitar": 3,
    "saxophone": 4
}

# 学習サイクル用の保存ディレクトリ作成
num_samples_per_class = 10  # 各クラスに対するトレーニングサンプル数制限
num_validation_samples = 10  # Validationのサンプル数
num_test_samples = 10        # Testのサンプル数
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
cycle_dir = f"C:/Users/klab/Desktop/back-matting-new/fine_tuned/InstrumentModel/{current_time}_samples_{num_samples_per_class}"
os.makedirs(cycle_dir, exist_ok=True)

# 共通関数: データローダー作成
def create_data_loader(annotation_file, dataset_type, num_samples=None):
    dataset = CustomDetectionDataset(
        data_dir=config["data_dir"],
        dataset_type=dataset_type,
        annotation_files=[annotation_file]
    )

    if len(dataset) == 0:
        print(f"Warning: No data available for {dataset_type}. Skipping.")
        return None

    if num_samples:
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        subset_indices = indices[:num_samples]
        dataset = Subset(dataset, subset_indices)

    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=(dataset_type == "train"))

# 各クラスごとにトレーニングとバリデーション
classes = ["human", "cello", "piano", "guitar", "saxophone"]
for class_name in classes:
    print(f"Training for class: {class_name}")

    # トレーニングデータローダー
    train_annotation_file = f"C:/Users/klab/Desktop/back-matting-new/data/train/train_annotation_{class_name}.csv"
    train_loader = create_data_loader(train_annotation_file, "train", num_samples_per_class)

    if train_loader is None:
        print(f"Skipping training for class: {class_name} due to lack of training data.")
        continue

    # クラス別トレーニングループ
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0

        # tqdmプログレスバーの設定
        with tqdm(total=len(train_loader), desc=f"Class: {class_name}, Epoch [{epoch+1}/{config['num_epochs']}]", unit="batch") as pbar:
            for batch in train_loader:
                if isinstance(batch, list) and len(batch) == 6:
                    images, masks, seg1, seg2, bbox, class_labels = batch
                else:
                    raise ValueError(f"Unexpected data format in train_loader: {batch}")

                # デバイスに転送
                images = images.to(device, dtype=torch.float)
                masks = masks[:, :1, :, :].to(device, dtype=torch.float)  # マスクを1チャンネルに変換

                # クラスラベルを数値に変換
                class_labels = torch.tensor([label_mapping[class_name] for _ in class_labels]).to(device)

                optimizer.zero_grad()

                # モデルの予測出力
                bbox_preds, class_preds = model(images, images, images, images)

                # クラス予測の形状調整
                batch_size, class_channels, width, height = class_preds.shape
                class_preds = class_preds.permute(0, 2, 3, 1).reshape(-1, class_channels)
                class_labels = class_labels.view(-1).repeat(width * height).long()

                # bbox_predsとmasksのサイズを一致させるためにリサイズ
                masks_resized = F.interpolate(masks, size=bbox_preds.shape[2:], mode="bilinear", align_corners=False)

                # 損失の計算
                loss_bbox = criterion_bbox(bbox_preds, masks_resized)
                loss_class = criterion_class(class_preds, class_labels)
                loss = loss_bbox + loss_class

                # 逆伝播と最適化
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # プログレスバーの更新
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

            # エポックごとの平均損失の表示
            avg_loss = running_loss / len(train_loader)
            print(f"Class: {class_name}, Epoch [{epoch+1}/{config['num_epochs']}], Average Loss: {avg_loss:.4f}")

    # クラスごとにモデルを保存
    model_save_path = os.path.join(cycle_dir, f"model_{class_name}_{current_time}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"モデルが保存されました for class {class_name}: {model_save_path}")

    # バリデーションデータローダー
    print(f"Validating for class: {class_name}")
    val_annotation_file = f"C:/Users/klab/Desktop/back-matting-new/data/validation/validation_annotation_{class_name}.csv"
    val_loader = create_data_loader(val_annotation_file, "validation", num_validation_samples)

    if val_loader is None:
        print(f"Skipping validation for class: {class_name}")
    else:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, list) and len(batch) == 6:
                    images, masks, seg1, seg2, bbox, class_labels = batch
                else:
                    raise ValueError(f"Unexpected data format in validation_loader: {batch}")

                images = images.to(device, dtype=torch.float)
                masks = masks[:, :1, :, :].to(device, dtype=torch.float)  # マスクを1チャンネルに変換
                class_labels = torch.tensor([label_mapping[class_name] for _ in class_labels]).to(device)

                bbox_preds, class_preds = model(images, images, images, images)

                masks_resized = F.interpolate(masks, size=bbox_preds.shape[2:], mode="bilinear", align_corners=False)
                batch_size, class_channels, width, height = class_preds.shape
                class_preds = class_preds.permute(0, 2, 3, 1).reshape(-1, class_channels)
                class_labels = class_labels.view(-1).repeat(width * height).long()

                loss_bbox = criterion_bbox(bbox_preds, masks_resized)
                loss_class = criterion_class(class_preds, class_labels)
                loss = loss_bbox + loss_class

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation for class {class_name}, Average Loss: {avg_val_loss:.4f}")

# テストデータローダーの設定
print("Testing model on test data")
test_annotation_file = f"C:/Users/klab/Desktop/back-matting-new/data/test/test_annotation_{class_name}.csv"
test_loader = create_data_loader(test_annotation_file, "test", num_test_samples)

# テスト結果保存ディレクトリの作成
test_output_base_dir = "C:/Users/klab/Desktop/back-matting-new/fine_tuned/in"
test_run_dir_name = f"{current_time}_train_{num_samples_per_class}"
test_results_dir = os.path.join(test_output_base_dir, test_run_dir_name)
os.makedirs(test_results_dir, exist_ok=True)

print(f"Test results will be saved in: {test_results_dir}")

# テストデータでの評価＆予測結果の保存
if test_loader is None:
    print("Skipping test evaluation due to lack of test data.")
else:
    model.eval()
    test_loss = 0.0
    predictions = []
    saved_image_count = 0  # 保存した画像の枚数をカウント
    max_images_to_save = 10  # 保存する画像の最大枚数

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if isinstance(batch, list) and len(batch) == 6:
                images, masks, seg1, seg2, bbox, class_labels = batch
            else:
                raise ValueError(f"Unexpected data format in test_loader: {batch}")

            images = images.to(device, dtype=torch.float)
            masks = masks[:, :1, :, :].to(device, dtype=torch.float)
            class_labels = class_labels.to(device)

            bbox_preds, class_preds = model(images, images, images, images)

            masks_resized = F.interpolate(masks, size=bbox_preds.shape[2:], mode="bilinear", align_corners=False)
            batch_size, class_channels, width, height = class_preds.shape
            class_preds = class_preds.permute(0, 2, 3, 1).reshape(-1, class_channels)
            class_labels = class_labels.view(-1).repeat(width * height).long()

            loss_bbox = criterion_bbox(bbox_preds, masks_resized)
            loss_class = criterion_class(class_preds, class_labels)
            loss = loss_bbox + loss_class

            test_loss += loss.item()

            # 保存用の結果を生成
            for i in range(len(images)):
                if saved_image_count >= max_images_to_save:
                    break  # 保存する画像が上限に達した場合は終了

                image_np = images[i].cpu().numpy().transpose(1, 2, 0)  # チャネルを移動
                image_np = (image_np * 255).astype(np.uint8)  # 正規化を戻す

                predicted_bbox = bbox_preds[i].cpu().numpy().flatten()
                predicted_class = torch.argmax(class_preds[i]).item()

                # 結果をリストに保存
                predictions.append({
                    "image_index": idx * batch_size + i,
                    "predicted_class": predicted_class,
                    "predicted_bbox": predicted_bbox.tolist()
                })

                # 画像に描画して保存
                plt.figure(figsize=(6, 6))
                plt.imshow(image_np)
                plt.gca().add_patch(plt.Rectangle(
                    (predicted_bbox[0], predicted_bbox[1]),
                    predicted_bbox[2] - predicted_bbox[0],
                    predicted_bbox[3] - predicted_bbox[1],
                    edgecolor='red', linewidth=2, fill=False
                ))
                plt.title(f"Predicted Class: {list(label_mapping.keys())[predicted_class]}")
                plt.axis('off')

                # 画像を保存
                result_image_path = os.path.join(test_results_dir, f"result_{idx}_{i}.png")
                plt.savefig(result_image_path)
                plt.close()

                saved_image_count += 1  # 保存した画像の数をカウント

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test data, Average Loss: {avg_test_loss:.4f}")

    # 結果をCSVで保存
    predictions_csv_path = os.path.join(test_results_dir, "predictions.csv")
    import pandas as pd
    pd.DataFrame(predictions).to_csv(predictions_csv_path, index=False)
    print(f"Test predictions saved in {predictions_csv_path}")
    print(f"Test result images saved in {test_results_dir}")
