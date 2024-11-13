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

# 各クラスごとにトレーニングとバリデーション
classes = ["human", "cello", "piano", "guitar", "saxophone"]
for class_name in classes:
    print(f"Training for class: {class_name}")

    # クラスごとのデータセットとデータローダーの設定
    annotation_file = f"C:/Users/klab/Desktop/back-matting-new/data/train/train_annotation_{class_name}.csv"
    train_dataset = CustomDetectionDataset(
        data_dir=config["data_dir"],
        dataset_type=config["dataset_type"]["train"],
        annotation_files=[annotation_file]
    )

    # サンプル数制限を適用
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    selected_indices = indices[:num_samples_per_class]
    train_subset = Subset(train_dataset, selected_indices)
    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)

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

    # 各クラスの学習終了ごとにバリデーション
    print(f"Validating for class: {class_name}")
    val_annotation_file = f"C:/Users/klab/Desktop/back-matting-new/data/validation/validation_annotation_{class_name}.csv"
    validation_dataset = CustomDetectionDataset(
        data_dir=config["data_dir"],
        dataset_type=config["dataset_type"]["validation"],
        annotation_files=[val_annotation_file]
    )

    if len(validation_dataset) == 0:
        print(f"Warning: No validation data available for class {class_name}. Skipping validation.")
    else:
        indices = np.arange(len(validation_dataset))
        np.random.shuffle(indices)
        selected_indices = indices[:num_validation_samples]
        validation_subset = Subset(validation_dataset, selected_indices)
        validation_loader = DataLoader(validation_subset, batch_size=config["batch_size"], shuffle=False)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
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

        avg_val_loss = val_loss / len(validation_loader)
        print(f"Validation for class {class_name}, Average Loss: {avg_val_loss:.4f}")

# テストデータで評価
print("Testing model on test data")
test_annotation_file = f"C:/Users/klab/Desktop/back-matting-new/data/test/test_annotation_{class_name}.csv"
test_dataset = CustomDetectionDataset(
    data_dir=config["data_dir"],
    dataset_type=config["dataset_type"]["test"],
    annotation_files=[test_annotation_file]  # クラスごとのテストアノテーションファイル
)

# テストデータが存在するか確認
if len(test_dataset) == 0:
    print("Warning: No test data available. Skipping test evaluation.")
else:
    indices = np.arange(len(test_dataset))
    np.random.shuffle(indices)
    selected_indices = indices[:num_test_samples]  # Testサンプル数制限
    test_subset = Subset(test_dataset, selected_indices)
    test_loader = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, list) and len(batch) == 6:
                images, masks, seg1, seg2, bbox, class_labels = batch
            else:
                raise ValueError(f"Unexpected data format in test_loader: {batch}")

            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)
            class_labels = class_labels.to(device)

            bbox_preds, class_preds = model(images, images, images, images)

            masks_resized = F.interpolate(masks, size=bbox_preds.shape[2:], mode="bilinear", align_corners=False)
            batch_size, class_channels, width, height = class_preds.shape
            class_preds = class_preds.permute(0, 2, 3, 1).reshape(-1, class_channels)
            class_labels = class_labels.view(-1).repeat(width * height).long()

            # 損失の計算
            loss_bbox = criterion_bbox(bbox_preds, masks_resized)
            loss_class = criterion_class(class_preds, class_labels)
            loss = loss_bbox + loss_class

            test_loss += loss.item()

    # テストデータの平均損失
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test data, Average Loss: {avg_test_loss:.4f}")

# 最終モデルの保存
final_model_save_path = os.path.join(cycle_dir, f"final_model_{current_time}.pth")
torch.save(model.state_dict(), final_model_save_path)
print(f"最終モデルが保存されました: {final_model_save_path}")
