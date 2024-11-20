import pandas as pd
from scripts.dataset import CustomDetectionDataset

from torch.utils.data import DataLoader

# アノテーションファイルのパス
annotation_files = {
    "train": "C:/Users/klab/Desktop/back-matting-new/data/train/train_annotation_human.csv",
    "validation": "C:/Users/klab/Desktop/back-matting-new/data/validation/validation_annotation_human.csv",
    "test": "C:/Users/klab/Desktop/back-matting-new/data/test/test_annotation_human.csv"
}

# **1. アノテーションファイルのフォーマット確認**
def check_annotation_file(file_path):
    print(f"Checking annotation file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("Columns:", df.columns)  # 列名の確認
        print("First 5 rows:\n", df.head())  # 先頭データの表示
        print("Missing values:\n", df.isnull().sum())  # 欠損値の確認
    except Exception as e:
        print(f"Error reading annotation file: {e}")

print("\n** Checking annotation files **")
for key, path in annotation_files.items():
    check_annotation_file(path)

# **2. データセットクラスの挙動確認**
def check_dataset(data_dir, dataset_type, annotation_file):
    print(f"\nChecking dataset for {dataset_type} using {annotation_file}")
    try:
        dataset = CustomDetectionDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            annotation_files=[annotation_file]
        )
        print(f"Dataset size: {len(dataset)}")

        # 最初の5件を確認
        for i in range(min(5, len(dataset))):
            data = dataset[i]
            print(f"Sample {i}: {data}")
    except Exception as e:
        print(f"Error initializing dataset: {e}")

print("\n** Checking datasets **")
check_dataset(
    data_dir="C:/Users/klab/Desktop/back-matting-new/data/",
    dataset_type="validation",
    annotation_file=annotation_files["validation"]
)

# **3. データローダーの挙動確認**
def check_dataloader(data_dir, dataset_type, annotation_file, batch_size=2):
    print(f"\nChecking dataloader for {dataset_type} using {annotation_file}")
    try:
        dataset = CustomDetectionDataset(
            data_dir=data_dir,
            dataset_type=dataset_type,
            annotation_files=[annotation_file]
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch_idx, batch in enumerate(data_loader):
            print(f"Batch {batch_idx}: {batch}")
            if batch_idx == 2:  # 最初の3バッチのみ確認
                break
    except Exception as e:
        print(f"Error initializing dataloader: {e}")

print("\n** Checking dataloader **")
check_dataloader(
    data_dir="C:/Users/klab/Desktop/back-matting-new/data/",
    dataset_type="validation",
    annotation_file=annotation_files["validation"]
)
