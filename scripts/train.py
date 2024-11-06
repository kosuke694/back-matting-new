import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from dataset import CustomDetectionDataset
from model import MyDetectionModel
from datetime import datetime
from tqdm import tqdm

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
    print(f"\nTraining for class: {class_name}")

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
        
        # tqdmを使用して進行状況バーを表示
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}") as pbar:
            for batch_idx, (images, masks, bboxes, class_labels) in enumerate(train_loader):
                images = images.to(device).float()
                masks = masks.to(device).float()
                bboxes = bboxes.to(device).float()
                class_labels = class_labels.to(device).long()

                optimizer.zero_grad()

                # モデルの予測出力
                bbox_preds, class_preds = model(images)

                # 損失の計算
                loss_bbox = criterion_bbox(bbox_preds, bboxes)
                
                # クラス予測とラベルをリシェイプ
                batch_size, num_classes, height, width = class_preds.size()
                class_preds = class_preds.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [batch_size * height * width, num_classes]
                class_labels = class_labels.expand(batch_size, height, width).reshape(-1)  # [batch_size * height * width]
                loss_class = criterion_class(class_preds, class_labels)

                # 逆伝播と最適化
                loss = (loss_bbox + loss_class).float()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # tqdmバーを更新し、残り時間を表示
                pbar.set_postfix(loss=running_loss / (batch_idx + 1), eta=int((time.time() - start_time) * (len(train_loader) - batch_idx - 1) / (batch_idx + 1)))
                pbar.update(1)

        avg_loss = running_loss / len(train_loader)
        print(f"\nClass: {class_name}, Epoch [{epoch+1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

    # 各クラスのトレーニング完了後、モデルを保存
    model_save_path = os.path.join(config["save_dir"], f"model_{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"モデルが保存されました for class {class_name}: {model_save_path}")

# 全クラスデータで最終エポックを実行（最終調整）
print("\nFinal Epoch: Training on a subset of all classes combined")
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
    
    # tqdmを使用して進行状況バーを表示
    with tqdm(total=len(final_train_loader), desc="Final Epoch") as pbar:
        for batch_idx, (images, masks, bboxes, class_labels) in enumerate(final_train_loader):
            images = images.to(device).float()
            masks = masks.to(device).float()
            bboxes = bboxes.to(device).float()
            class_labels = class_labels.to(device).long()

            optimizer.zero_grad()

            # モデルの予測出力
            bbox_preds, class_preds = model(images)

            # 損失の計算
            loss_bbox = criterion_bbox(bbox_preds, bboxes)
            
            # クラス予測とラベルをリシェイプ
            batch_size, num_classes, height, width = class_preds.size()
            class_preds = class_preds.permute(0, 2, 3, 1).reshape(-1, num_classes)  # [batch_size * height * width, num_classes]
            class_labels = class_labels.expand(batch_size, height, width).reshape(-1)  # [batch_size * height * width]
            loss_class = criterion_class(class_preds, class_labels)

            # 逆伝播と最適化
            loss = (loss_bbox + loss_class).float()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # tqdmバーを更新し、残り時間を表示
            pbar.set_postfix(loss=running_loss / (batch_idx + 1), eta=int((time.time() - start_time) * (len(final_train_loader) - batch_idx - 1) / (batch_idx + 1)))
            pbar.update(1)

    avg_loss = running_loss / len(final_train_loader)
    print(f"\nFinal Epoch, Combined Classes (subset), Loss: {avg_loss:.4f}")

# 最終エポック後に最終モデルを保存
final_model_save_path = os.path.join(config["save_dir"], f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
torch.save(model.state_dict(), final_model_save_path)
print(f"最終モデルが保存されました: {final_model_save_path}")
