import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import MyModel  # ユーザーが定義した MyModel をインポート
from dataset import CustomDataset  # ユーザーが定義したデータセット
import os

def load_pretrained_model(model, pretrained_model_path, device):
    """
    事前学習済みモデルの状態辞書をロードし、"module." プレフィックスの有無を自動的に調整し、
    必要に応じてキー名を変換してモデルにロードします。
    """
    print(f"事前学習済みモデルをロード中: {pretrained_model_path}")

    # 事前学習済みモデルの状態辞書をロード
    state_dict = torch.load(pretrained_model_path, map_location=device)

    # 現在のモデルのキーを取得
    model_keys = set(model.state_dict().keys())

    # 新しい状態辞書を作成
    new_state_dict = {}

    for key, value in state_dict.items():
        # プレフィックス "module." の追加/削除を自動調整
        if key.startswith('module.') and not any(k.startswith('module.') for k in model_keys):
            new_key = key.replace('module.', '')
        elif not key.startswith('module.') and any(k.startswith('module.') for k in model_keys):
            new_key = 'module.' + key
        else:
            new_key = key

        # サイズの不一致がある層をスキップ
        if new_key in model.state_dict() and model.state_dict()[new_key].shape == value.shape:
            new_state_dict[new_key] = value
        else:
            print(f"Skipping loading parameter: {new_key} due to size mismatch.")

    # 新しい状態辞書をモデルにロード
    model.load_state_dict(new_state_dict, strict=False)
    print("事前学習済みモデルのロードが完了しました。")

def train(model, dataloader, criterion, optimizer, device, num_epochs=10):
    """
    モデルをトレーニングするための関数
    Args:
        model (nn.Module): トレーニングするモデル
        dataloader (DataLoader): トレーニングデータのデータローダ
        criterion (nn.Module): 損失関数
        optimizer (torch.optim.Optimizer): オプティマイザ
        device (torch.device): 使用するデバイス（GPU/CPU）
        num_epochs (int): エポック数
    """
    model.train()  # モデルをトレーニングモードに設定

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 順伝播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 逆伝播とオプティマイザステップ
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    # 設定ファイルや引数から情報を取得する部分（例）
    config = {
        "pretrained_model_path": "pre-trained_Models/real-fixed-cam/netG_epoch_12.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 16,
        "learning_rate": 0.001,
        "num_epochs": 10,
        "data_dir": "path/to/your/dataset"  # データセットのディレクトリパスを指定
    }

    device = torch.device(config["device"])

    # モデルのインスタンスを作成（ユーザーが定義する MyModel クラス）
    model = MyModel().to(device)

    # 事前学習済みモデルのロード
    load_pretrained_model(model, config["pretrained_model_path"], device)

    # データセットの準備（CustomDataset）
    train_dataset = CustomDataset(data_dir=config["data_dir"], transform=None)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # 損失関数とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()  # 適切な損失関数に変更する必要あり
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # モデルのトレーニング
    print("モデルのトレーニングを開始します...")
    train(model, train_loader, criterion, optimizer, device, num_epochs=config["num_epochs"])

    print("モデルのトレーニングが完了しました。")
