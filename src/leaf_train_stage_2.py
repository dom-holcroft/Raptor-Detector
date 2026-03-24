import os
import sys
import subprocess
from pathlib import Path

def main():
    # 1. Define all paths
    root_dir = Path(os.getcwd())
    leaf_yolo_dir = root_dir / "external" / "LEAF-YOLO"
    
    # We grab the best weights from the Stage 2 run you are doing right now!
    weights_path = leaf_yolo_dir / "runs" / "train" / "stage1_bird_pretrained" / "weights" / "best.pt"
    
    # The new YAML files we just created in the root folder
    data_yaml = root_dir / "raptor_data.yaml"
    hyp_yaml = root_dir / "hyp.raptor_detector.yaml"

    # 2. Safety Check
    if not weights_path.exists():
        print(f"❌ ERROR: Cannot find Stage 2 weights at {weights_path}")
        print("Wait for Stage 2 to generate best.pt, or check the folder name!")
        return

    # 3. Build the YOLO training command
    command = [
        sys.executable, "train.py",
        "--weights", str(weights_path),
        "--cfg", "",  # Leave blank, it will infer from the weights file
        "--data", str(data_yaml),
        "--hyp", str(hyp_yaml),
        "--epochs", "150",           # 150 epochs for the final stage
        "--batch-size", "32",        # Safe batch size for RTX 3070
        "--img-size", "640", "640",
        "--device", "0",
        "--name", "stage3_raptor_final",
    ]

    print("🚀 Launching Stage 3 Training from Root Directory...")
    print(f"🧠 Starting Brain: {weights_path.name}")
    print(f"📂 Dataset: {data_yaml.name}")
    print("-" * 50)

    # 4. Execute the command *inside* the LEAF-YOLO folder
    try:
        subprocess.run(command, cwd=str(leaf_yolo_dir), check=True)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training crashed with error code: {e.returncode}")

if __name__ == "__main__":
    main()