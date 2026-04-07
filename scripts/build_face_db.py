import os
import pickle
import cv2
import numpy as np

# -------------------------- 路径配置（和你的项目目录对齐） --------------------------
# 人脸照片存放目录（face_db 文件夹）
FACE_IMAGES_DIR = "../face_db"
# 生成的人脸库保存路径
DB_SAVE_PATH = "../face_db/encodings.pkl"
# OpenCV人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 初始化人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()


def build_face_database():
    faces = []
    labels = []
    label_map = {}  # 用于把姓名映射为数字标签

    # 遍历每个人的文件夹
    for person_name in os.listdir(FACE_IMAGES_DIR):
        person_dir = os.path.join(FACE_IMAGES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"正在处理 {person_name} 的照片...")
        # 为每个人分配一个唯一的数字标签
        if person_name not in label_map:
            label_map[person_name] = len(label_map)
        label = label_map[person_name]

        # 遍历该人所有的照片
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            # 只处理图片文件
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 读取图片并转换为灰度图
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 检测人脸
            detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # 提取人脸区域并保存
            for (x, y, w, h) in detected_faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (200, 200))  # 统一尺寸
                faces.append(face_roi)
                labels.append(label)

    # 训练识别器并保存模型
    recognizer.train(faces, np.array(labels))
    recognizer.save("../face_db/recognizer.yml")
    # 保存姓名映射关系
    with open(DB_SAVE_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"✅ 人脸库构建完成！共录入 {len(label_map)} 人")


if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("../face_db", exist_ok=True)
    build_face_database()
