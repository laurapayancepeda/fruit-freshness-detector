from ultralytics import YOLO

# YOLOv8 nano
model = YOLO("yolov8n.pt")

# CPU (using local computer)
model.train(
    data=r"C:\Users\laura\OneDrive\Documents\CV-PROJECT\Dataset\data.yaml",
    epochs=10,
    imgsz=320,
    batch=8,
    device="cpu",
    workers=2,
    augment=True,
)

# probably need to do more epochs and with gpu
print("Training complete. Model saved in runs/detect/train/weights/best.pt")
