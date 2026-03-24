import os
import cv2
import json
from pathlib import Path

# Setup your directories
videos_dir = Path("videos")
annotations_dir = Path("annotations")
manifest_path = "dataset_manifest.json"

dataset_index = []

# Loop through every mp4 in the videos folder
for video_path in videos_dir.glob("*.mp4"):
    video_name = video_path.stem
    
    # Locate the corresponding XML 
    # (Assuming your structure is: annotations/201421521/annotations.xml)
    xml_path = annotations_dir / video_name / "annotations.xml"
    
    if not xml_path.exists():
        print(f"⚠️ Warning: No annotations found for {video_name}")
        continue

    # Quickly peek at the video to get the total frame count
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Apply your specific video logic
    if video_name == "201855901":
        # Cut out the last 5 frames
        dataset_index.append({
            "clip_id": f"{video_name}_trimmed",
            "video_path": str(video_path),
            "xml_path": str(xml_path),
            "start_frame": 0,
            "end_frame": total_frames - 6 # -6 ensures we drop the last 5
        })
        
    elif video_name == "622466935":
        # Split into two logical clips
        dataset_index.append({
            "clip_id": f"{video_name}_part1",
            "video_path": str(video_path),
            "xml_path": str(xml_path),
            "start_frame": 0,
            "end_frame": 1880
        })
        dataset_index.append({
            "clip_id": f"{video_name}_part2",
            "video_path": str(video_path),
            "xml_path": str(xml_path),
            "start_frame": 4685,
            "end_frame": 5868
        })
        
    else:
        # Default behavior: use the whole video
        dataset_index.append({
            "clip_id": video_name,
            "video_path": str(video_path),
            "xml_path": str(xml_path),
            "start_frame": 0,
            "end_frame": total_frames - 1
        })

# Save the map to a JSON file
with open(manifest_path, "w") as f:
    json.dump(dataset_index, f, indent=4)

print(f"✅ Success! Manifest created with {len(dataset_index)} dataset clips.")