import os
import shutil
from pathlib import Path

# Source base paths
image_base = Path("images")
label_base = Path("labels")

# Destination paths
output_image_train = Path("datasets/screws/images/train")
output_label_train = Path("datasets/screws/labels/train")
output_image_train.mkdir(parents=True, exist_ok=True)
output_label_train.mkdir(parents=True, exist_ok=True)

# Supported image extensions
image_exts = [".jpg", ".jpeg", ".png"]

# Loop through each screwtype
for screwtype in ["Screwtype1", "Screwtype2", "Screwtype3", "Screwtype4"]:
    image_train_folder = image_base / screwtype / "train"
    label_train_folder = label_base / screwtype / "train"

    for image_file in image_train_folder.iterdir():
        if image_file.suffix.lower() in image_exts:
            label_file = label_train_folder / f"{image_file.stem}.txt"

            # Output filenames (keep original names, may overwrite if duplicated!)
            out_img = output_image_train / image_file.name
            out_lbl = output_label_train / label_file.name

            shutil.copy(image_file, out_img)
            if label_file.exists():
                shutil.copy(label_file, out_lbl)
