import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import time
from dataset import CustomDetectionDataset

# モデル定義
class MyDetectionModel(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(MyDetectionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.bbox_head = nn.Linear(256 * 4 * 4, 4)  # バウンディングボックスの予測
        self.class_head = nn.Linear(256 * 4 * 4, output_nc)  # クラスの予測

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)  # 特徴マップを平坦化
        bbox_preds = self.bbox_head(x)
        class_preds = self.class_head(x)
        return bbox_preds, class_preds

# 設定ファイルの読み込み
with open("C:/Users/klab/Desktop/back-matting-new/configs/train_config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの初期化
model = MyDetectionModel(input_nc=3, output_nc=5).to(device)

# 事前学習モデルの読み込み（存在する場合）
if "pretrained_model_path" in config and os.path.exists(config["pretrained_model_path"]):
    model.load_state_dict(torch.load(config["pretrained_model_path"], map_location=device))

# 損失関数と最適化手法
criterion_bbox = nn.MSELoss()            # バウンディングボックス用損失関数
criterion_class = nn.CrossEntropyLoss()  # クラス分類用損失関数
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# データセットとデータローダーの設定
train_dataset = CustomDetectionDataset(
    dataset_type=config["dataset_types"]["train"],
    annotation_files=config["train_annotation_files"],
    data_dir=config["data_dir"]
)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

val_dataset = CustomDetectionDataset(
    dataset_type=config["dataset_types"]["validation"],
    annotation_files=config["val_annotation_files"],
    data_dir=config["data_dir"]
)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

test_dataset = CustomDetectionDataset(
    dataset_type=config["dataset_types"]["test"],
    annotation_files=config["test_annotation_files"],
    data_dir=config["data_dir"]
)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# トレーニングループ
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
        loss_class = criterion_class(class_preds, class_labels.view(-1).long())
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
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {avg_loss:.4f}")

    # バリデーションの実行
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks, bboxes, class_labels in val_loader:
            images, masks, bboxes, class_labels = images.to(device), masks.to(device), bboxes.to(device), class_labels.to(device)

            bbox_preds, class_preds = model(images)

            loss_bbox = criterion_bbox(bbox_preds, bboxes)
            loss_class = criterion_class(class_preds, class_labels.view(-1).long())
            val_loss += (loss_bbox + loss_class).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss after Epoch [{epoch + 1}/{config['num_epochs']}]: {avg_val_loss:.4f}")

# モデルの保存
model_save_path = os.path.join("C:/Users/klab/Desktop/back-matting-new/fine_tuned/InstrumentModel", 
                               f"model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"モデルが保存されました: {model_save_path}")

# テストの実行
print("Testing on test dataset")
model.eval()
test_loss = 0.0
test_results = []
with torch.no_grad():
    for images, masks, bboxes, class_labels in test_loader:
        images, masks, bboxes, class_labels = images.to(device), masks.to(device), bboxes.to(device), class_labels.to(device)

        bbox_preds, class_preds = model(images)

        loss_bbox = criterion_bbox(bbox_preds, bboxes)
        loss_class = criterion_class(class_preds, class_labels.view(-1).long())
        total_loss = loss_bbox + loss_class
        test_loss += total_loss.item()
        test_results.append((total_loss.item(), loss_bbox.item(), loss_class.item()))

avg_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {avg_test_loss:.4f}")

# テスト結果の保存
test_results_path = "C:/Users/klab/Desktop/back-matting-new/fine_tuned/InstrumentModel/test_results.txt"
os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
with open(test_results_path, "w") as f:
    f.write("Test Losses (Total Loss, BBox Loss, Class Loss):\n")
    for result in test_results:
        f.write(f"{result}\n")
    f.write(f"\nAverage Test Loss: {avg_test_loss:.4f}\n")

print(f"テスト結果が保存されました: {test_results_path}")
