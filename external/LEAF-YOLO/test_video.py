import cv2
import torch
import numpy as np

# Import core YOLO components
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

def main():
    weights = 'runs/train/stage1_bird_pretrained/weights/best.pt'
    source = r'C:\Users\Dominic\Documents\Raptor-Detector\videos\468137.mp4'
    output_path = 'test_output.mp4'

    print("🚀 1. Loading PyTorch Model on CPU...")
    device = torch.device('cpu')
    model = attempt_load(weights, map_location=device)
    model.eval()  # Set model to inference mode
    print("✅ Model loaded successfully!")

    print(f"🎬 2. Opening Video: {source}")
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # mp4v is the most reliable video codec for OpenCV on Windows
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("🧠 3. Starting Inference Loop...")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Print frame 1 so we know the model didn't crash, then every 10 frames
        if frame_count == 1 or frame_count % 10 == 0:
            print(f"Processing frame {frame_count}/{total_frames}...")

        # --- PREPROCESS ---
        # 1. Resize the 1080p frame down to YOLO's 640x640 with padding
        img = letterbox(frame, 640, stride=32)[0]
        # 2. Convert BGR to RGB, and move channels to the front (HWC to CHW)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # 3. Convert to PyTorch tensor and normalize pixels to 0.0 - 1.0
        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # Add the batch dimension

        # --- INFERENCE ---
        with torch.no_grad():
            pred = model(img)[0]

        # --- POSTPROCESS ---
        # Apply Non-Max Suppression (Confidence threshold = 0.25, IoU threshold = 0.45)
        pred = non_max_suppression(pred, 0.1, 0.45, classes=None, agnostic=False)

        # --- DRAW BOXES ---
        for i, det in enumerate(pred):  # Iterate through detections
            if len(det):
                # Rescale the 640x640 boxes back up to the original 1920x1080 video size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    # Draw using raw OpenCV
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(frame, c1, c2, (0, 255, 0), 2)
                    cv2.putText(frame, f'Bird {conf:.2f}', (c1[0], c1[1] - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save the drawn frame to the new video
        out.write(frame)

    cap.release()
    out.release()
    print(f"🎉 DONE! Video saved locally as {output_path}")

if __name__ == '__main__':
    main()