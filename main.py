from ultralytics import YOLO
import torch
import gc

def train_yolo_model():
    model = YOLO('yolov8s.pt')  # Try 's' or 'm' version

    model.train(
        data='datasets/screws/screw.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        lr0=0.001,
        weight_decay=0.0005,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        val=False,
        patience=10,
    )

    metrics = model.val()
    print(f"✅ Final mAP@0.5: {metrics.box.map50:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print("🚀 Starting YOLOv8 improved training...")
    train_yolo_model()
