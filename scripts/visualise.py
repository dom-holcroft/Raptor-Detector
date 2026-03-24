from ultralytics import YOLO
import os

# 1. Load your best trained model
model = YOLO("runs/detect/p2_experiment/weights/best.pt")

# 2. Path to your videos folder
video_folder = "videos"
# Pick a specific video to test (change this to any filename in your folder)
test_video = os.path.join(video_folder, "643092607.mp4") 

# 3. Run inference with the tracker turned on
# We use 'show=True' to watch it live, and 'save=True' to export a result video
results = model.track(
    source=test_video, 
    conf=0.3,      # Only show boxes the model is >30% sure about
    iou=0.5,       # Smooth out overlapping boxes
    save=True,     # Save the result to runs/detect/track/
    tracker="botsort.yaml" # Turn on the tracker we discussed for wingbeat data
)

print(f"Test complete. Check 'runs/detect/predict' for the output video!")