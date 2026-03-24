import os
import sys
import subprocess
from pathlib import Path

def main():
    # 1. Define exact absolute paths
    project_root = Path(__file__).parent.parent.absolute()
    leaf_dir = project_root / "external" / "LEAF-YOLO"
    
    # CHANGED: Point this to the YAML file that defines your new SAHI dataset paths
    data_yaml = project_root / "bird_detection.yaml" 
    
    # NEW: Define the path to your custom Adam + NWD hyperparameters
    hyp_yaml = project_root / "hyp.small_bird_detector.yaml"
    
    # Define where you saved the GitHub weights
    weights_path = leaf_dir / "cfg" / "LEAF-YOLO" / "leaf-sizen" / "weights" / "best.pt"
    
    if not leaf_dir.exists():
        print(f"❌ Error: Could not find the LEAF-YOLO directory at {leaf_dir}")
        sys.exit(1)

    if not hyp_yaml.exists():
        print(f"❌ Error: Could not find the hyperparameter file at {hyp_yaml}")
        sys.exit(1)

    # Safety check to make sure the weights are actually there
    if not weights_path.exists():
        print(f"❌ Error: Could not find the pre-trained weights at {weights_path}")
        sys.exit(1)

    # 2. Build the training command
    command = [
        sys.executable, "train.py",
        "--workers", "8",           # Speeds up data loading 
        "--device", "0",            # Uses your primary NVIDIA GPU
        "--batch-size", "32",       # 32 fits perfectly on your 8GB RTX 3070
        "--epochs", "150",          # Keeping it 150 for a fair comparison
        
        "--data", str(data_yaml),
        "--img", "640", "640",      # Your SAHI patches are 640x640, so this is perfect
        
        "--cfg", "cfg/LEAF-YOLO/leaf-sizen.yaml", 
        "--weights", str(weights_path), 
        
        # NEW: Inject your custom NWD and Augmentation hyperparameters
        "--hyp", str(hyp_yaml),
        
        # NEW: Force LEAF-YOLO to use the Adam(W) Optimizer instead of SGD
        "--adam",
        
        # CHANGED: Name updated to reflect exactly what this run is
        "--name", "stage1_bird_sahi_adam" 
    ]

    print("🚀 Launching LEAF-YOLO Stage 1 (SAHI + Adam + NWD) Training...")
    print(f"📂 Working Directory: {leaf_dir}")
    print(f"📊 Dataset Config: {data_yaml}")
    print(f"⚙️  Hyperparameters: {hyp_yaml}")
    print(f"🧠 Loaded Weights: {weights_path}")
    
    try:
        # cwd=leaf_dir is what makes the magic happen. 
        subprocess.run(command, cwd=str(leaf_dir), check=True)
        print("✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Training crashed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted manually.")

if __name__ == "__main__":
    main()