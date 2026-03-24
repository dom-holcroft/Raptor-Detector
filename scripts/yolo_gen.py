import os
import cv2
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# --- Configuration ---
MANIFEST_PATH = "dataset_manifest.json"
OUTPUT_DIR = Path("yolo_dataset")
FRAME_STEP = 10 # Extract 1 frame every 10 frames to prevent overfitting
CLASSES = {"Raptor": 0, "Non-Raptor": 1} # Adjust if your CVAT labels differ slightly
TRAIN_RATIO = 0.8 # 80% of clips for training, 20% for testing/validation

def setup_directories():
    for split in ['train', 'val']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def parse_cvat_xml(xml_path):
    """Parses CVAT 1.1 Video XML and returns a dict mapping frame_id to boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get video dimensions from XML metadata
    orig_width = float(root.find('.//original_size/width').text) if root.find('.//original_size/width') is not None else None
    orig_height = float(root.find('.//original_size/height').text) if root.find('.//original_size/height') is not None else None
    
    frame_boxes = {}
    
    for track in root.findall('track'):
        label = track.get('label')
        if label not in CLASSES:
            continue
            
        class_id = CLASSES[label]
        
        for box in track.findall('box'):
            # Ignore boxes where the object is marked as outside the frame
            if box.get('outside') == '1':
                continue
                
            frame_id = int(box.get('frame'))
            xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
            xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
            
            if frame_id not in frame_boxes:
                frame_boxes[frame_id] = []
                
            frame_boxes[frame_id].append((class_id, xtl, ytl, xbr, ybr))
            
    return frame_boxes, orig_width, orig_height

def main():
    setup_directories()
    
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
        
    # Shuffle and split clips (Video-level split prevents data leakage)
    random.shuffle(manifest)
    split_idx = int(len(manifest) * TRAIN_RATIO)
    train_clips = manifest[:split_idx]
    
    for clip in manifest:
        split = 'train' if clip in train_clips else 'val'
        video_path = clip['video_path']
        xml_path = clip['xml_path']
        clip_name = clip['clip_id']
        
        print(f"Processing {clip_name} -> {split} set...")
        
        frame_boxes, w, h = parse_cvat_xml(xml_path)
        
        cap = cv2.VideoCapture(video_path)
        # Fallback if XML doesn't have dimensions
        if w is None: w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if h is None: h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        current_frame = clip['start_frame']
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        while current_frame <= clip['end_frame']:
            ret, frame = cap.read()
            if not ret: break
            
            # Only process if this frame is in our step AND has annotations
            if (current_frame % FRAME_STEP == 0) and (current_frame in frame_boxes):
                img_filename = f"{clip_name}_frame_{current_frame}.jpg"
                lbl_filename = f"{clip_name}_frame_{current_frame}.txt"
                
                img_path = OUTPUT_DIR / 'images' / split / img_filename
                lbl_path = OUTPUT_DIR / 'labels' / split / lbl_filename
                
                # Save Image
                cv2.imwrite(str(img_path), frame)
                
                # Calculate YOLO format & save Label
                with open(lbl_path, "w") as f_lbl:
                    for (cls_id, xtl, ytl, xbr, ybr) in frame_boxes[current_frame]:
                        # Math to convert to YOLO percentages
                        x_center = ((xtl + xbr) / 2) / w
                        y_center = ((ytl + ybr) / 2) / h
                        box_width = (xbr - xtl) / w
                        box_height = (ybr - ytl) / h
                        
                        f_lbl.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            current_frame += 1
            
        cap.release()
        
    print("\n✅ Dataset conversion complete!")

if __name__ == "__main__":
    main()