import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import MyModel  # あなたのモデルに合わせて変更
from datetime import datetime
import matplotlib.pyplot as plt

# 設定
config = {
    "data_dir": "data/input",  # 入力画像ディレクトリ
    "mask_dir": "data/masks",  # マスク画像ディレクトリ
    "batch_size": 4,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "pretrained_model_path": "pre-trained_Models/real-fixed-cam/netG_epoch_12.pth",
    "save_dir": "fine_tuned/InstrumentModel",  # モデルの保存ディレクトリ
}

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの定義とロード
model = MyModel().to(device)
if os.path.exists(config["pretrained_model_path"]):
    print(f"事前学習済みモデルをロード中: {config['pretrained_model_path']}")
    state_dict = torch.load(config["pretrained_model_path"], map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("事前学習済みモデルのロードが完了しました。")
else:
    print("事前学習済みモデルが見つかりませんでした。新しいモデルでトレーニングを開始します。")

# データセットとデータローダーの準備
transform = None  # 必要に応じてトランスフォームを追加
train_dataset = CustomDataset(data_dir=config["data_dir"], mask_dir=config["mask_dir"], transform=transform)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

# 損失関数と最適化手法
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# トレーニングループ
losses = []
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 平均損失の計算
    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"エポック {epoch + 1} が完了しました。平均損失: {avg_loss}")

# モデルの保存
if not os.path.exists(config["save_dir"]):
    os.makedirs(config["save_dir"])

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_save_path = os.path.join(config["save_dir"], f"fine_tuned_model_{timestamp}.pth")
torch.save(model.state_dict(), model_save_path)
print(f"トレーニング済みモデルが保存されました: {model_save_path}")

# 損失関数の可視化
plt.figure()
plt.plot(range(1, config["num_epochs"] + 1), losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
loss_plot_path = os.path.join(config["save_dir"], f"training_loss_{timestamp}.png")
plt.savefig(loss_plot_path)
plt.close()
print(f"損失関数のグラフが保存されました: {loss_plot_path}")

# 予測結果の可視化
model.eval()
sample_images, sample_masks = next(iter(train_loader))
sample_images = sample_images.to(device)
sample_masks = sample_masks.to(device)
with torch.no_grad():
    outputs = model(sample_images)

# 結果をプロット
fig, axes = plt.subplots(3, config["batch_size"], figsize=(12, 6))
for i in range(config["batch_size"]):
    axes[0, i].imshow(sample_images[i].cpu().permute(1, 2, 0))  # 入力画像
    axes[1, i].imshow(sample_masks[i].cpu().squeeze(), cmap='gray')  # 実際のマスク
    # 予測されたマスクを表示
    axes[2, i].imshow(outputs[i].cpu().squeeze()[0], cmap='gray')  # 予測されたマスクの1チャンネルを表示


for ax in axes.flat:
    ax.axis('off')

output_plot_path = os.path.join(config["save_dir"], f"output_comparison_{timestamp}.png")
plt.savefig(output_plot_path)
plt.close()
print(f"予測結果の比較グラフが保存されました: {output_plot_path}")
