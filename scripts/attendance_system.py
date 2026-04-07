import os
import cv2
import pickle
import csv
import datetime
from ultralytics import YOLO

# -------------------------- 路径配置（已按你的实际路径修正） --------------------------
# 🔥 关键：修正了路径，和你的文件夹结构完全匹配！
YOLO_MODEL_PATH = "../runs/detect/runs/train/yolov8_face2/weights/best.pt"

RECOGNIZER_PATH = "../face_db/recognizer.yml"
DB_PATH = "../face_db/encodings.pkl"
ATTENDANCE_LOG = "../attendance_log/attendance.csv"
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 120  # 识别置信度，越低越严格

# -------------------------- 初始化模型 --------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 安全检查：确保文件存在
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"❌ 错误：找不到 YOLO 模型文件！\n路径：{YOLO_MODEL_PATH}")
    exit(1)

if not os.path.exists(RECOGNIZER_PATH):
    print(f"❌ 错误：找不到人脸库文件！\n请先运行 build_face_db.py")
    exit(1)

# 加载模型
recognizer.read(RECOGNIZER_PATH)
with open(DB_PATH, "rb") as f:
    label_map = pickle.load(f)
id_to_name = {v: k for k, v in label_map.items()}


# -------------------------- 签到记录函数 --------------------------
def log_attendance(name):
    """记录签到信息到CSV"""
    os.makedirs(os.path.dirname(ATTENDANCE_LOG), exist_ok=True)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # 检查今日是否已签到
    already_signed = False
    try:
        with open(ATTENDANCE_LOG, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == name and row[1] == date_str:
                    already_signed = True
                    break
    except FileNotFoundError:
        # 文件不存在则写入表头
        with open(ATTENDANCE_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["姓名", "日期", "签到时间"])

    if not already_signed:
        with open(ATTENDANCE_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])
        print(f"✅ {name} 签到成功！")
    else:
        print(f"ℹ️ {name} 今日已签到")


# -------------------------- 主程序 --------------------------
def main():
    # 加载 YOLO 模型
    yolo = YOLO(YOLO_MODEL_PATH)
    print("✅ 模型加载成功！正在开启摄像头...")

    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("❌ 错误：无法打开摄像头")
        return

    print("摄像头已开启！按 'q' 退出系统")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break

        # YOLO 检测人脸
        results = yolo(frame, conf=0.6)
        for result in results:
            for box in result.boxes:
                # 获取坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 截取人脸区域
                face_roi = frame[y1:y2, x1:x2]

                # 识别
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_resized = cv2.resize(gray, (200, 200))
                label, confidence = recognizer.predict(gray_resized)

                # 判断结果
                if confidence < CONFIDENCE_THRESHOLD:
                    name = id_to_name.get(label, "未知人员")
                    log_attendance(name)  # 自动签到
                else:
                    name = "未知人员"

                # 绘制方框和文字（显示置信度）
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.1f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("Face Attendance System", frame)
        # 按 q 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")


if __name__ == "__main__":
    main()
