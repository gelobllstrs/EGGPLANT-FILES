from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("No GPU detected, using CPU.")
        device = "cpu"

    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="C:/Users/gelob/Desktop/yolov8 seg dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,   # safest on Windows
        name="eggplant_seg"
    )

if __name__ == "__main__":
    freeze_support()
    main()