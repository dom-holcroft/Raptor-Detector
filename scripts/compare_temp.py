import os
import cv2
import shutil
import random
import sys
import subprocess
from pathlib import Path
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
ROOT = Path(os.getcwd())
TEST_IMAGES = ROOT / "data/raptor_dataset/images/test"
TEST_LABELS = ROOT / "data/raptor_dataset/labels/test"  # <-- Added for Ground Truth
COMPARE_DIR = ROOT / "runs" / "compare"
INPUT_DIR = COMPARE_DIR / "inputs"
OUTPUT_DIR = COMPARE_DIR / "side_by_side"

# Use forward slashes for the command line to prevent Windows path bugs
LEAF_WEIGHTS = str(ROOT / "external/LEAF-YOLO/runs/train/stage3_raptor_final5/weights/best.pt").replace("\\", "/")
V8_WEIGHTS = str(ROOT / "runs/detect/v8_p2_retrain_fair/weights/best.pt").replace("\\", "/")
LEAF_DIR = ROOT / "external/LEAF-YOLO"

CLASSES = {0: "Raptor", 1: "Non-Raptor"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)} 

def setup_folders():
    if COMPARE_DIR.exists(): shutil.rmtree(COMPARE_DIR)
    for p in [INPUT_DIR, OUTPUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def draw_boxes(img_path, labels_file, title):
    """Draws boxes onto the image. Handles both Ground Truth (no conf) and Predictions (with conf)."""
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    
    cv2.rectangle(img, (0, 0), (w, 50), (0,0,0), -1)
    cv2.putText(img, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    
    if labels_file and labels_file.exists():
        with open(labels_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5: # Need at least cls, x, y, w, h
                    cls_id = int(parts[0])
                    x_center, y_center, box_w, box_h = map(float, parts[1:5])
                    
                    # Check if it has a confidence score (Ground Truth files don't have this)
                    conf = float(parts[5]) if len(parts) > 5 else None
                    
                    x1 = int((x_center - box_w / 2) * w)
                    y1 = int((y_center - box_h / 2) * h)
                    x2 = int((x_center + box_w / 2) * w)
                    y2 = int((y_center + box_h / 2) * h)
                    
                    color = COLORS.get(cls_id, (255,255,0))
                    label = f"{CLASSES.get(cls_id, 'Unknown')}"
                    if conf is not None:
                        label += f" {conf:.2f}"
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x1, y1 - 25), (x1 + tw, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return img

def main():
    setup_folders()
    print("📁 Grabbing 5 random test images...")
    
    all_imgs = list(TEST_IMAGES.glob("*.jpg")) + list(TEST_IMAGES.glob("*.png"))
    if not all_imgs:
        print(f"❌ ERROR: No images found in {TEST_IMAGES}")
        return
        
    test_imgs = random.sample(all_imgs, min(5, len(all_imgs)))
    for img in test_imgs: shutil.copy(img, INPUT_DIR / img.name)

    # --- RUN YOLOv8 INFERENCE ---
    print("\n🧠 Running YOLOv8-P2...")
    v8_model = YOLO(V8_WEIGHTS)
    v8_model.predict(
        source=str(INPUT_DIR), save_txt=True, save_conf=True, 
        project=str(COMPARE_DIR), name="v8_predict", conf=0.25, verbose=False, exist_ok=True
    )
    v8_label_dir = COMPARE_DIR / "v8_predict" / "labels"

    # --- RUN LEAF-YOLO INFERENCE ---
    print("\n🌿 Running LEAF-YOLO...")
    leaf_cmd = [
        sys.executable, "detect.py",
        "--weights", LEAF_WEIGHTS, 
        "--source", str(INPUT_DIR).replace("\\", "/"),
        "--conf-thres", "0.25", 
        "--save-txt", 
        "--save-conf", 
        "--nosave",
        "--project", str(COMPARE_DIR).replace("\\", "/"), 
        "--name", "leaf_predict",
        "--exist-ok" # <-- This stops it from creating leaf_predict2, leaf_predict3, etc.
    ]
    
    # We removed the silencer so you can see if LEAF-YOLO prints an error
    try:
        subprocess.run(leaf_cmd, cwd=str(LEAF_DIR), check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ LEAF-YOLO crashed during inference! Check the terminal output above.")
        return

    leaf_label_dir = COMPARE_DIR / "leaf_predict" / "labels"

    # --- STITCH IMAGES TOGETHER ---
    print("\n🎨 Generating 3-Way Side-by-Side Comparisons...")
    for img_path in INPUT_DIR.glob("*.*"):
        gt_txt = TEST_LABELS / f"{img_path.stem}.txt"
        v8_txt = v8_label_dir / f"{img_path.stem}.txt"
        leaf_txt = leaf_label_dir / f"{img_path.stem}.txt"
        
        img_gt = draw_boxes(img_path, gt_txt if gt_txt.exists() else None, "1. Ground Truth")
        img_leaf = draw_boxes(img_path, leaf_txt if leaf_txt.exists() else None, "2. LEAF-YOLO")
        img_v8 = draw_boxes(img_path, v8_txt if v8_txt.exists() else None, "3. YOLOv8 + P2")
        
        # Stitch all three horizontally
        stitched = cv2.hconcat([img_gt, img_leaf, img_v8])
        
        # Resize to fit on screen without breaking aspect ratio
        scale_ratio = 1920 / stitched.shape[1]
        new_height = int(stitched.shape[0] * scale_ratio)
        stitched_small = cv2.resize(stitched, (1920, new_height))
        
        cv2.imwrite(str(OUTPUT_DIR / f"COMPARE_{img_path.name}"), stitched_small)

    print(f"\n🎉 DONE! Open '{OUTPUT_DIR.absolute()}' to view the results!")

if __name__ == "__main__":
    main()