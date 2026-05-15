import kagglehub
import shutil

# Download to kaggle cache
download_path = kagglehub.dataset_download(
    "andrewmvd/ocular-disease-recognition-odir5k"
)

# Copy into your project root
shutil.copytree(
    download_path,
    "./ocular-disease-recognition-odir5k",
    dirs_exist_ok=True
)

print("Dataset downloaded to project root")