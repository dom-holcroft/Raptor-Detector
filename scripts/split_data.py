import os
import random
import shutil
from pathlib import Path
from sahi.slicing import slice_coco
from ultralytics.data.converter import convert_coco

def build_bird_dataset():
    project_root = Path(__file__).parent.parent.absolute()
    target_empty_ratio = 0.10 

    datasets = [
        {"name": "drone2021_train", "split": "train", "images": "data/drone2021/images/", "annotations": "data/drone2021/annotations/split_train_coco.json"},
        {"name": "sod4bird_train", "split": "train", "images": "data/mva2023_sod4bird_train/images/", "annotations": "data/mva2023_sod4bird_train/annotations/split_train_coco.json"},
        {"name": "drone2021_val", "split": "val", "images": "data/drone2021/images/", "annotations": "data/drone2021/annotations/split_val_coco.json"},
        {"name": "sod4bird_val", "split": "val", "images": "data/mva2023_sod4bird_train/images/", "annotations": "data/mva2023_sod4bird_train/annotations/split_val_coco.json"}
    ]

    for ds in datasets:
        print(f"\n🚀 Processing {ds['name']}...")
        temp_output = project_root / f"temp_{ds['name']}"
        
        final_images_dir = project_root / f"data/small_bird_dataset/images/{ds['split']}"
        final_labels_dir = project_root / f"data/small_bird_dataset/labels/{ds['split']}"
        final_images_dir.mkdir(parents=True, exist_ok=True)
        final_labels_dir.mkdir(parents=True, exist_ok=True)

        # --- STEP 1: RESCUE LOGIC ---
        if temp_output.exists():
            print(f"📦 Temp folder found! Skipping slicing for {ds['name']}...")
        else:
            print(f"🔪 Slicing {ds['name']} (This will take time)...")
            slice_coco(
                coco_annotation_file_path=ds["annotations"],
                image_dir=ds["images"],
                output_coco_annotation_file_name=ds["name"],
                output_dir=str(temp_output),
                slice_height=640, slice_width=640,
            )

        # --- STEP 2: CONVERT (WITH FIX) ---
        # cls91to80=False prevents the NoneType crash!
        convert_coco(labels_dir=str(temp_output), use_segments=False, cls91to80=False)

        # --- STEP 3: FIND LABELS ---
        potential_label_dirs = list(project_root.glob("coco_converted*"))
        if not potential_label_dirs:
            print(f"❌ Could not find converted labels for {ds['name']}")
            continue
        
        # SAHI adds _coco to the output JSON filename
        latest_label_dir = max(potential_label_dirs, key=os.path.getmtime) / "labels" / f"{ds['name']}_coco"

        # --- STEP 4: BULLETPROOF IMAGE FINDER ---
        print(f"📊 Analyzing backgrounds for {ds['split']}...")
        empty_files = []
        non_empty_files = []

        # Find all images inside temp folder regardless of where SAHI put them
        all_images = [p for p in temp_output.rglob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        img_dict = {img.stem: img for img in all_images}

        for txt_path in latest_label_dir.glob("*.txt"):
            if txt_path.stem in img_dict:
                img_path = img_dict[txt_path.stem]
                if txt_path.stat().st_size == 0:
                    empty_files.append((txt_path, img_path))
                else:
                    non_empty_files.append((txt_path, img_path))

        # --- STEP 5: RATIO & MOVE ---
        if ds['split'] == "train":
            num_empty = int(len(non_empty_files) * (target_empty_ratio / (1.0 - target_empty_ratio)))
            selected_empty = random.sample(empty_files, min(num_empty, len(empty_files)))
        else:
            selected_empty = empty_files

        files_to_move = non_empty_files + selected_empty
        print(f"📦 Moving {len(files_to_move)} files to {ds['split']}...")
        
        for txt_path, img_path in files_to_move:
            shutil.copy(str(img_path), final_images_dir / f"{ds['name']}_{img_path.name}")
            shutil.copy(str(txt_path), final_labels_dir / f"{ds['name']}_{txt_path.name}")

        # --- STEP 6: CLEANUP ---
        shutil.rmtree(temp_output)
        if latest_label_dir.parent.parent.exists():
            shutil.rmtree(latest_label_dir.parent.parent)

    print("\n🎉 RECOVERY COMPLETE. Check data/small_bird_dataset/")

if __name__ == "__main__":
    build_bird_dataset()