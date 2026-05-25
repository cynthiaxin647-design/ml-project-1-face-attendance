# 机器学习项目一：基于 YOLOv8 的人脸检测与识别签到系统

## 项目简介

  本项目综合运用 YOLOv8（人脸检测）和 OpenCV（人脸识别）技术，实现一个完整的智能签到系统，具备以下核心功能：

1. 检测人脸 — 使用 YOLOv8 实时检测画面中的所有人脸
2. 识别并报出姓名 — 将检测到的人脸与已注册的人脸库匹配，显示姓名
3. 统计签到总人数 — 实时显示已签到人数
4. 不重复计数 — 同一人多次出现只记录一次
5. convert_wider_face.py（转换脚本）— 把 WIDER FACE 数据集自动转成 YOLO 格式

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境
conda create -n face-attend python=3.10 -y
conda activate face-attend

# 安装 PyTorch（CPU版）
pip install torch torchvision torchaudio

# 安装项目依赖
pip install ultralytics          # YOLOv8
pip install opencv-contrib-python   # 图像处理+人脸识别
pip install numpy pandas         # 数据处理
```
### 2. 准备人脸库

在 `face_db/` 目录下为每个人创建以姓名命名的文件夹，放入 3~5 张清晰正面照片:

```
face_db/
├── 张三/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── 李四/
│   ├── 01.jpg
│   └── 02.jpg
```

然后构建特征库:

```bash
python scripts/build_face_db.py
```

### 3. 训练 YOLO 人脸检测模型

准备好数据集后，运行:

```bash
python scripts/train_yolo.py
```

### 4. 运行签到系统

```bash
python scripts/attendance_system.py
```

- 按 q 退出，自动导出签到记录 CSV
- 按 s 手动截图保存

### 5. 图片检测（可选）

```bash
python scripts/detect_image.py --image test.jpg
python scripts/detect_image.py --dir test_images/
```

## 项目结构

```
ml-project-1-face-attendance/
├── dataset/                 # YOLO 训练数据集
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   ├── labels/val/
│   └── face_data.yaml
├── face_db/                 # 人脸识别特征库
│   ├── 张三/                # 每人一个文件夹
│   ├── 李四/
│   └── encodings.pkl       # 自动生成的特征编码文件
├── scripts/
│   ├── train_yolo.py       # Step 1: 训练 YOLO 模型
│   ├── build_face_db.py    # Step 2: 构建人脸特征库
│   ├── attendance_system.py # Step 3: 核心签到系统
│   └── detect_image.py     # 图片检测模式
├── runs/                    # YOLO 训练输出
├── attendance_log/          # 签到记录输出
├── requirements.txt
└── README.md
```

## 技术架构

```
摄像头/图片 → YOLOv8检测人脸 → 裁剪人脸区域 → OpenCV LBPH 特征提取
                                              ↓
CSV签到记录 ← 去重计数 ← 签到记录 ← 姓名匹配 ← 与人脸库比对
```

## 参考资源

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [WIDER FACE 数据集](http://shuoyang1213.me/WIDERFACE/)
- [Roboflow](https://roboflow.com/)
```

