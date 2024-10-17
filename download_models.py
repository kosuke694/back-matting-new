import gdown
import os

# モデルファイルの保存ディレクトリ
if not os.path.exists('models'):
    os.makedirs('models')

# Google DriveのファイルIDと保存先のファイル名を設定
file_ids = {
    'netG_epoch_12.pth': '1G9MKkv9SRLLhR-7YFbecYxXs64_MmEeD',
    'netG_epoch_44.pth': '1FM0TuO8V7hrgHYEQMmI6DDhme1n1UYoc',
    'net_epoch_64.pth': '1S9jZ99JjM-ncatL1zQJ2Z8puLN8IHIq5',
    'deeplabv3_pascal_trainval_2018_01_04.tar.gz': '1Af0NRtlrut_T_VqCMTPtb_whBzcCHtct'
}

# Google Driveからモデルファイルをダウンロード
for file_name, file_id in file_ids.items():
    output = f'models/{file_name}'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        print(f'{file_name} has been downloaded!')
    else:
        print(f'{file_name} already exists.')
