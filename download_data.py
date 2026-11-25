from pathlib import Path
import shutil
import kagglehub

# Tải dữ liệu từ Kaggle
cache_path = Path(kagglehub.dataset_download("msambare/fer2013"))

# Chép toàn bộ sang thư mục data\raw của bạn
target_dir = Path(r"D:\Download\Computor-Vision_FinalLab\data\raw")
target_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)

print("XONG! Du lieu da nam tai:", target_dir)