import os
import random
import shutil
from pathlib import Path

# Paths
base_dir = Path("data/yolo_dataset")
train_images = base_dir / "images/train"
train_labels = base_dir / "labels/train"

val_images = base_dir / "images/val"
val_labels = base_dir / "labels/val"

# Create Val folders
val_images.mkdir(parents=True, exist_ok=True)
val_labels.mkdir(parents=True, exist_ok=True)

# Get all images
all_images = list(train_images.glob("*.*"))
split_ratio = 0.10 # 10% for validation
val_count = int(len(all_images) * split_ratio)

print(f"Total images found: {len(all_images)}")
print(f"Moving {val_count} images to validation set...")

# Randomly select images for validation
val_sample = random.sample(all_images, val_count)

for img_path in val_sample:
    # Move the image
    shutil.move(str(img_path), val_images / img_path.name)
    
    # Find and move the matching label
    label_name = img_path.stem + ".txt"
    label_path = train_labels / label_name
    
    if label_path.exists():
        shutil.move(str(label_path), val_labels / label_name)

print("✅ Validation split complete! Ready for LEAF-YOLO training.")