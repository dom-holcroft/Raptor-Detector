import yaml
import torch
from pathlib import Path

# Import directly from your LEAF-YOLO codebase
from utils.datasets import create_dataloader
from utils.plots import plot_images

def main():
    print("🚀 Starting Data Pipeline Diagnostic...")

    # 1. Define paths (Update these if your files are in a different spot)
    data_yaml_path = '../../bird_detection.yaml'  # Path to your dataset config
    hyp_yaml_path = 'data/hyp.scratch.p5.yaml'    # Path to YOLO hyperparameters

    # Load dataset config to find the training images
    with open(data_yaml_path, 'r') as f:
        data_dict = yaml.safe_load(f)
    train_path = data_dict['train']

    # Load hyperparameters (this controls the mosaic, copy-paste, etc.)
    with open(hyp_yaml_path, 'r') as f:
        hyp = yaml.safe_load(f)

    # 2. Mock the 'opt' arguments that train.py usually provides
    class MockOpt:
        single_cls = True
        cache_images = False
        rect = False
        world_size = 1
        workers = 0  # Set to 0 to avoid Windows multiprocessing crashes during a quick test
        image_weights = False
        quad = False

    opt = MockOpt()

    # 3. Initialize the Dataloader exactly as train.py does
    print(f"📂 Loading images from: {train_path}")
    dataloader, dataset = create_dataloader(
        path=train_path,
        imgsz=640,
        batch_size=4,        # We only want 4 images for a quick test
        stride=32,
        opt=opt,
        hyp=hyp,
        augment=True,        # 🔥 CRITICAL: This turns on Mosaic, Copy-Paste, and random zooming!
        cache=False,
        rank=-1,
        workers=0,
        prefix='diagnostic: '
    )

    # 4. Pull exactly ONE batch of data
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        print(f"📦 Successfully loaded 1 batch. Tensor shape: {imgs.shape}")
        
        # 5. Use YOLO's built-in plotting tool to draw the boxes
        output_filename = 'test_pipeline_output.jpg'
        
        # plot_images automatically un-normalizes the tensors and draws the bounding boxes
        plot_images(imgs, targets, paths, output_filename, max_size=1280, max_subplots=4)
        
        print(f"✅ Diagnostic complete! Open '{output_filename}' to inspect the boxes.")
        break  # Stop after the first batch

if __name__ == '__main__':
    main()