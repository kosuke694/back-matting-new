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
import time

# 設定ファイルの読み込み
with open("C:/Users/klab/Desktop/back-matting-new/configs/train_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの初期化
model = MyDetectionModel(input_nc=3, output_nc=1, num_classes=5).to(device)

# 損失関数と最適化手法
criterion_bbox = nn.MSELoss()            # バウンディングボックス回帰用損失関数
criterion_class = nn.CrossEntropyLoss()  # クラス分類用損失関数
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# 各クラスごとにトレーニング
classes = ["human", "cello", "piano", "guitar", "saxophone"]
for class_name in classes:
    print(f"Training for class: {class_name}")

    # クラスごとのデータセットとデータローダーの設定
    annotation_file = f"data/train/annotations_{class_name}.csv"
    train_dataset = CustomDetectionDataset(
        dataset_type=config["dataset_type"],
        annotation_files=[annotation_file],
        data_dir=config["data_dir"]
    )
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # クラス別トレーニングループ
    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for batch_idx, (images, masks, bboxes, class_labels) in enumerate(train_loader):
            images, masks, bboxes, class_labels = images.to(device), masks.to(device), bboxes.to(device), class_labels.to(device)

            optimizer.zero_grad()

            # モデルの予測出力
            bbox_preds, class_preds = model(images)

            # 損失の計算
            loss_bbox = criterion_bbox(bbox_preds, bboxes)
            loss_class = criterion_class(class_preds, class_labels.view(-1).long())  # class_labelsの形状を調整
            loss = loss_bbox + loss_class

            # 逆伝播と最適化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # 進捗表示
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / (batch_idx + 1) * len(train_loader)
            remaining_time = estimated_total_time - elapsed_time
            print(f"Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f} - "
                  f"Elapsed: {elapsed_time:.2f}s - Remaining: {remaining_time:.2f}s")

        avg_loss = running_loss / len(train_loader)
        print(f"Class: {class_name}, Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

    # 各クラスのトレーニング完了後、モデルを保存
    model_save_path = os.path.join("C:/Users/klab/Desktop/back-matting-new/fine_tuned/InstrumentModel", 
                                   f"model_{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"モデルが保存されました for class {class_name}: {model_save_path}")

# 全クラスデータで最終エポックを実行（最終調整）
print("Final Epoch: Training on a subset of all classes combined")
annotation_files = [f"data/train/annotations_{class_name}.csv" for class_name in classes]
final_train_dataset = CustomDetectionDataset(
    dataset_type=config["dataset_type"],
    annotation_files=annotation_files,
    data_dir=config["data_dir"]
)

# データサンプル数の制限（例えばクラスごとに100サンプルのみ使用）
num_samples_per_class = 100  # 各クラスのサンプル数を指定

# ランダムにサブセットを作成（全クラスから指定したサンプル数を抽出）
all_indices = np.arange(len(final_train_dataset))
np.random.shuffle(all_indices)
selected_indices = all_indices[:num_samples_per_class * len(classes)]

# サブセットデータローダーの作成
final_train_subset = Subset(final_train_dataset, selected_indices)
final_train_loader = DataLoader(final_train_subset, batch_size=config["batch_size"], shuffle=True)

# 最終エポックのトレーニングループ
for epoch in range(1):  # 最終エポック1回
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (images, masks, bboxes, class_labels) in enumerate(final_train_loader):
        images, masks, bboxes, class_labels = images.to(device), masks.to(device), bboxes.to(device), class_labels.to(device)

        optimizer.zero_grad()

        # モデルの予測出力
        bbox_preds, class_preds = model(images)

        # 損失の計算
        loss_bbox = criterion_bbox(bbox_preds, bboxes)
        loss_class = criterion_class(class_preds, class_labels.view(-1).long())  # class_labelsの形状を調整
        loss = loss_bbox + loss_class

        # 逆伝播と最適化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # 進捗表示
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (batch_idx + 1) * len(final_train_loader)
        remaining_time = estimated_total_time - elapsed_time
        print(f"Batch [{batch_idx + 1}/{len(final_train_loader)}] - Loss: {loss.item():.4f} - "
              f"Elapsed: {elapsed_time:.2f}s - Remaining: {remaining_time:.2f}s")

    avg_loss = running_loss / len(final_train_loader)
    print(f"Final Epoch, Combined Classes (subset), Loss: {avg_loss:.4f}")

# 最終エポック後に最終モデルを保存
final_model_save_path = os.path.join("C:/Users/klab/Desktop/back-matting-new/fine_tuned/InstrumentModel", 
                                     f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
torch.save(model.state_dict(), final_model_save_path)
print(f"最終モデルが保存されました: {final_model_save_path}")
