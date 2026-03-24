from ultralytics import YOLO


def train_basic():

    # Load a pre-trained, lightweight PyTorch model (YOLOv8 Nano)
    model = YOLO("yolov8n.pt") 

    # Fine-tune the model on your extracted dataset
    results = model.train(data="raptor_data.yaml", epochs=50, imgsz=640)

    print("Training complete! Your weights are saved in the 'runs/detect' folder.")

def train_pt():
    model = YOLO("yolov8n-p2.yaml") 
    
    # We can still load the pre-trained weights to give it a head start!
    model.load("yolov8n.pt")

    # Train it!
    results = model.train(data="raptor_data.yaml", epochs=50, imgsz=640, project="BirdDetector", name="P2_Experiment")
# This acts as a lock. Only the main script can pass this point, 
# the background workers will be stopped here!
if __name__ == '__main__':
    train_pt()
    

