import torch
from collections import OrderedDict
from model import ResnetConditionHR  # ResnetConditionHRが定義されているファイルからインポート

# 事前学習済みの重みファイルをロード
# 正しいファイルパスに置き換え
state_dict = torch.load("C:/Users/klab/Desktop/back-matting-new/pre-trained_Models/real-fixed-cam/netG_epoch_12.pth", map_location="cpu")


# 新しいstate_dictを作成し、'module.'プレフィックスを削除
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k  # 'module.'のプレフィックスを削除
    new_state_dict[name] = v

# モデルの初期化（ここでは仮にResnetConditionHRモデルとします）
model = ResnetConditionHR(input_nc=[3, 3, 3, 3], output_nc=4)  # 適切なモデルを使用してください
model_state_dict = model.state_dict()

# Missing keysとUnexpected keysを比較しながら取得
missing_keys = []
unexpected_keys = []

# モデルに存在するが、事前学習済み重みにないキー（missing keys）
for key in model_state_dict.keys():
    if key not in new_state_dict:
        missing_keys.append(key)

# 事前学習済み重みに存在するが、モデルにないキー（unexpected keys）
for key in new_state_dict.keys():
    if key not in model_state_dict:
        unexpected_keys.append(key)

# 総キー数、missing keysとunexpected keysの数を計算
total_keys = len(model_state_dict)
missing_keys_count = len(missing_keys)
unexpected_keys_count = len(unexpected_keys)
matched_keys_count = total_keys - missing_keys_count

# 比率を計算
missing_keys_ratio = (missing_keys_count / total_keys) * 100
unexpected_keys_ratio = (unexpected_keys_count / total_keys) * 100
matched_keys_ratio = (matched_keys_count / total_keys) * 100

# 結果を出力
print(f"Total keys in model: {total_keys}")
print(f"Matched keys: {matched_keys_count} ({matched_keys_ratio:.2f}%)")
print(f"Missing keys: {missing_keys_count} ({missing_keys_ratio:.2f}%)")
print(f"Unexpected keys: {unexpected_keys_count} ({unexpected_keys_ratio:.2f}%)")

# 詳細情報（オプション）
print("\nMissing keys:", missing_keys)
print("\nUnexpected keys:", unexpected_keys)
