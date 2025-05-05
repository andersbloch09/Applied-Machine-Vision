import os
import random
import shutil

# Paths to your dataset
base_path = "./datasets/screws"
images_train_path = os.path.join(base_path, "images/train")
images_val_path = os.path.join(base_path, "images/val")
labels_train_path = os.path.join(base_path, "labels/train")
labels_val_path = os.path.join(base_path, "labels/val")

# Create validation directories if they don't exist
os.makedirs(images_val_path, exist_ok=True)
os.makedirs(labels_val_path, exist_ok=True)

# Set the validation split ratio
val_ratio = 0.2

# Get all image files in the train directory
image_files = [f for f in os.listdir(images_train_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)  # Shuffle to ensure randomness

# Calculate the number of files to move
num_val = int(len(image_files) * val_ratio)

# Move images and their corresponding labels
for image_file in image_files[:num_val]:
    # Move the image
    src_image = os.path.join(images_train_path, image_file)
    dest_image = os.path.join(images_val_path, image_file)
    shutil.move(src_image, dest_image)

    # Move the corresponding label
    label_file = image_file.rsplit('.', 1)[0] + '.txt'  # Replace image extension with .txt
    src_label = os.path.join(labels_train_path, label_file)
    dest_label = os.path.join(labels_val_path, label_file)
    if os.path.exists(src_label):  # Ensure the label file exists before moving
        shutil.move(src_label, dest_label)

print("✅ Dataset split completed!")