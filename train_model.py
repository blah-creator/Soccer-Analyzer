from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="soccer_dataset/data.yaml",
    epochs=70,
    imgsz=640,
)
