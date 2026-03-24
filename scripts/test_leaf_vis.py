import os
import random
import sys
import subprocess
from pathlib import Path

# --- 1. Paths ---
ROOT_DIR = Path(os.getcwd())
TEST_IMAGES_DIR = ROOT_DIR / "data/raptor_dataset/images/test"
VIDEOS_DIR = ROOT_DIR / "videos"
LEAF_DIR = ROOT_DIR / "external" / "LEAF-YOLO"
WEIGHTS_PATH = LEAF_DIR / "runs" / "train" / "stage3_raptor_final5" / "weights" / "best.pt"

def get_test_video():
    """Scans the test images folder to reverse-engineer which videos are in the test split."""
    if not TEST_IMAGES_DIR.exists():
        print("❌ ERROR: Test images directory not found.")
        return None
        
    # Get all jpgs in the test folder
    test_frames = list(TEST_IMAGES_DIR.glob("*.jpg"))
    if not test_frames:
        print("❌ ERROR: No images found in the test folder.")
        return None
        
    # Extract the unique video IDs (e.g., from "643092607_frame_120.jpg" -> "643092607")
    test_vid_ids = list(set([f.stem.split("_frame_")[0] for f in test_frames]))
    
    # Pick a random video ID from the test set
    chosen_id = random.choice(test_vid_ids)
    print(f"🎲 Randomly selected Test Video ID: {chosen_id}")
    
    # Find the actual .mp4 file in the videos folder
    possible_paths = [VIDEOS_DIR / f"{chosen_id}.mp4", VIDEOS_DIR / f"ML{chosen_id}.mp4"]
    for p in possible_paths:
        if p.exists():
            return p
            
    print(f"❌ ERROR: Could not find the raw .mp4 file for {chosen_id} in {VIDEOS_DIR}")
    return None

def main():
    print("🔍 Setting up LEAF-YOLO Video Inference...")
    
    if not WEIGHTS_PATH.exists():
        print(f"❌ ERROR: Weights not found at {WEIGHTS_PATH}. Wait for training to save best.pt!")
        return

    test_video_path = get_test_video()
    if not test_video_path:
        return
        
    print(f"🎬 Target Video: {test_video_path.name}")
    
    # --- 2. Build the LEAF-YOLO detect.py command ---
    command = [
        sys.executable, "detect.py",
        "--weights", str(WEIGHTS_PATH),
        "--source", str(test_video_path),
        "--conf-thres", "0.30",    # Only show boxes with >30% confidence
        "--iou-thres", "0.45",     # NMS threshold to merge overlapping boxes
        "--img-size", "640",       # Inference resolution
        "--device", "0",           # Use GPU
        "--name", "test_video_out" # Output folder name
    ]
    
    print("\n🚀 Launching LEAF-YOLO Tracker...")
    print("-" * 50)
    
    # Run the command inside the LEAF-YOLO directory
    try:
        subprocess.run(command, cwd=str(LEAF_DIR), check=True)
        
        # The output will be saved in runs/detect/test_video_out
        out_folder = LEAF_DIR / "runs" / "detect" / "test_video_out"
        print("-" * 50)
        print(f"✅ Inference Complete!")
        print(f"📂 Open this folder to watch your labeled video: {out_folder.absolute()}")
        
    except KeyboardInterrupt:
        print("\n🛑 Inference interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference crashed with error code: {e.returncode}")

if __name__ == "__main__":
    main()