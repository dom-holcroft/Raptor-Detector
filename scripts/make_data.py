import os
import cv2
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# --- 1. CONFIGURATION ---
VIDEOS_DIR = Path("videos")
XML_PATH = Path("annotations/annotations.xml")
OUTPUT_DIR = Path("data/raptor_dataset")
FRAME_STEP = 3

YOLO_CLASSES = {"Raptor": 0, "Non-Raptor": 1}

SPECIES_GROUPS = {
    "Red_Kite": ["643574202", "608966902", "201466721", "201882221", "637403145", "631235527", "639621542", "639498597", "201898741", "637490538"],
    "Hen_Harrier": ["634656042", "634589541", "201490091", "457429561", "201421521"],
    "White_tailed_Eagle": ["622466935", "201492471", "201358681", "615683900", "201767701"],
    "Eurasian_Kestrel": ["637490548", "201492091", "201520471", "201885581", "451625301"],
    "Golden_Eagle": ["631990594", "613970661", "215841801", "481140", "215842691"],
    "Carrion_Crow": ["639698395", "639658930", "639621545", "201371651", "646703208"],
    "Herring_Gull": ["201471011", "614466527", "201817721", "468205", "468137"],
    "Cormorant": ["638431558", "636788178", "633162868", "631557603", "630419434"],
    "Wood_Pigeon": ["630107428", "629990957", "201499621", "201473281", "201855901"],
    "Swifts": ["643481638", "643092607", "643092595", "643092585", "642020001"]
}

def setup_directories():
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)

def parse_project_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    tasks = {}
    current_global_offset = 0
    
    for task in root.findall('.//task'):
        task_id = task.find('id').text
        source_name = task.find('source').text
        base_id = source_name.replace('.mp4', '').replace('ML', '').strip()
        size = int(task.find('size').text)
        
        tasks[task_id] = {'vid_id': base_id, 'offset': current_global_offset}
        current_global_offset += size
        
    video_boxes = {}
    for track in root.findall('.//track'):
        task_id = track.get('task_id')
        if task_id not in tasks: continue
            
        vid_id = tasks[task_id]['vid_id']
        offset = tasks[task_id]['offset']
        label = track.get('label')
        
        if label not in YOLO_CLASSES: continue
        class_id = YOLO_CLASSES[label]
        
        if vid_id not in video_boxes: video_boxes[vid_id] = {}
            
        for box in track.findall('box'):
            if box.get('outside') == '1': continue
            local_frame = int(box.get('frame')) - offset
            if local_frame < 0: continue
                
            xtl, ytl = float(box.get('xtl')), float(box.get('ytl'))
            xbr, ybr = float(box.get('xbr')), float(box.get('ybr'))
            
            if local_frame not in video_boxes[vid_id]:
                video_boxes[vid_id][local_frame] = []
            video_boxes[vid_id][local_frame].append((class_id, xtl, ytl, xbr, ybr))
            
    return video_boxes

def find_file(base_id, directory, extension):
    paths_to_try = [directory / f"{base_id}{extension}", directory / f"ML{base_id}{extension}"]
    for p in paths_to_try:
        if p.exists(): return p
    return None

def main():
    setup_directories()
    video_boxes = parse_project_xml(XML_PATH)
    
    split_manifest = {'train': [], 'val': [], 'test': []}
    for species, vid_list in SPECIES_GROUPS.items():
        random.shuffle(vid_list)
        total = len(vid_list)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9) 
        split_manifest['train'].extend(vid_list[:train_end])
        split_manifest['val'].extend(vid_list[train_end:val_end])
        split_manifest['test'].extend(vid_list[val_end:])

    total_frames_saved = 0
    
    for split_name, video_ids in split_manifest.items():
        print(f"\n🚀 Processing {split_name.upper()} set...")
        
        for vid_id in video_ids:
            vid_path = find_file(vid_id, VIDEOS_DIR, ".mp4")
            if not vid_path: continue
            if vid_id not in video_boxes: continue
                
            vid_boxes = video_boxes[vid_id]
            
            cap = cv2.VideoCapture(str(vid_path))
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # --- CUSTOM EXCEPTION LIMITS ---
            # Default: Entire video
            allowed_ranges = [(0, total_frames - 1)] 
            
            if vid_id == "201855901":
                allowed_ranges = [(0, total_frames - 6)]
            elif vid_id == "622466935":
                allowed_ranges = [(0, 1880), (4685, 5868)]
            # -------------------------------
            
            frame_count = 0
            saved_for_video = 0
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # Check if current frame is inside the allowed ranges
                in_range = any(start <= frame_count <= end for start, end in allowed_ranges)
                
                if in_range and (frame_count % FRAME_STEP == 0) and (frame_count in vid_boxes):
                    img_name = f"{vid_id}_frame_{frame_count}.jpg"
                    lbl_name = f"{vid_id}_frame_{frame_count}.txt"
                    
                    cv2.imwrite(str(OUTPUT_DIR / 'images' / split_name / img_name), frame)
                    
                    with open(OUTPUT_DIR / 'labels' / split_name / lbl_name, "w") as f:
                        for (cls_id, xtl, ytl, xbr, ybr) in vid_boxes[frame_count]:
                            x_center = ((xtl + xbr) / 2) / w
                            y_center = ((ytl + ybr) / 2) / h
                            box_w = (xbr - xtl) / w
                            box_h = (ybr - ytl) / h
                            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                            
                    saved_for_video += 1
                    total_frames_saved += 1
                    
                frame_count += 1
            cap.release()
            print(f"  └─ {vid_id}: Extracted {saved_for_video} frames.")

    print(f"\n🎉 DATASET COMPLETE! Extracted {total_frames_saved} images.")

if __name__ == "__main__":
    main()