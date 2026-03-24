import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # 1. Define paths
    root_dir = Path(os.getcwd())
    data_yaml = root_dir / "raptor_data.yaml"
    model_yaml = root_dir / "yolov8n-p2.yaml" 

    # 2. Safety Checks
    if not data_yaml.exists() or not model_yaml.exists():
        print("❌ ERROR: Cannot find data.yaml or model.yaml")
        return

    print("🚀 Initializing YOLOv8n with custom P2 Head AND Chaos Engine...")
    model = YOLO(str(model_yaml)).load("yolov8n.pt")

    # 3. Train the model with Explicit Hyperparameters
    model.train(
        data=str(data_yaml),
        epochs=150,
        batch=24,
        patience=25,
        imgsz=640,
        device=0,           
        # project="runs/detect",
        name="v8_p2_retrain_fair_pretrained",
        exist_ok=True,
        workers=8,
        
        # --- OPTIMIZER SETTINGS ---
        lr0=0.001,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # --- THE CHAOS ENGINE (AUGMENTATIONS) ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.9,         # The critical 90% scaling factor
        shear=0.0,
        perspective=0.0005, # Simulates angled flight
        flipud=0.0,         # No upside-down birds
        fliplr=0.5,         # 50% left/right mirror
        mosaic=1.0,         # 100% chance to smash 4 images together
        mixup=0.1,          # 10% chance to blend mosaics
        copy_paste=0.3      # 30% chance to copy/paste objects
    )
    
    print("\n✅ Fair Training Complete! Saved in runs/detect/v8_p2_retrain_fair/weights/best.pt")

if __name__ == "__main__":
    main()