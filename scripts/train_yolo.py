from ultralytics import YOLO
import os

# 打印当前工作目录
print("当前工作目录:", os.getcwd())

# 加载模型（用 yolov8n.pt）
model = YOLO("yolov8n.pt")

# 开始训练
results = model.train(
    data="../dataset/face_data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project="runs",
    name="train/yolov8_face",
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    augment=True,
    save=True,
    save_period=5,
    val=True,
    plots=True
)

print("训练完成！结果已保存到 runs/train/yolov8_face/")
