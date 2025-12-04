from ultralytics import YOLO
import os

# Path to your dataset (update as needed)
  # Roboflow auto-generates this; edit if needed:
# Example content for data.yaml:
# train: ./dataset/train/images
# val: ./dataset/valid/images
# nc: 1  # Number of classes
# names: ['weeds']
'''
# Load pre-trained YOLOv8 nano model
model = YOLO('yolov8n.pt')  # Or 'yolov8s.pt' for small/balanced

# Train the model
results = model.train(
    data=data_yaml_path,
    epochs=50,              # Adjust based on dataset size
    imgsz=640,              # Image size (paddy fields often need 640x640)
    batch=16,               # Batch size (reduce if low VRAM)
    name='paddy_weed_yolo', # Output folder name
    device=0 if os.system('nvidia-smi') == 0 else 'cpu',  # Auto GPU/CPU
    patience=10,            # Early stopping
    save=True,              # Save checkpoints
    plots=True              # Generate training plots
)

# Export model (optional, for deployment)
model.export(format='onnx')  # For inference on edge devices
'''
# Use YOLOv8n or YOLOv10n – both run great on CPU with this setup
model = YOLO("yolov8n.pt")        # ← safest & most tested on CPU
# model = YOLO("yolov10n.pt")     # ← uncomment if you want slightly faster inference later

model.train(
    data=r"data.yaml",
    epochs=5,
    imgsz=416,              # small = fast on CPU
    batch=8,                # ← 8 or 16 is perfect for CPU (don’t go higher)
    cache=True,             # 'ram' also works if you have 16GB+ RAM
    device='cpu',           # ← THIS IS THE FIX

    # Super-fast CPU settings
    workers=4,              # number of CPU cores to use for loading data
    patience=15,
    name="paddy_weed_CPU_FAST",
    project="runs/train",

    # Disable everything heavy
    amp=False,              # mixed precision doesn't help much on CPU
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.3,
    fliplr=0.5,
    flipud=0.0
)

print("Training finished! Model saved in runs/train/paddy_weed_CPU_FAST/")