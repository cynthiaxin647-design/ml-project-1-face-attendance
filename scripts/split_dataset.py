import os
import random
import shutil

# 比例：20% 划为验证集
split_ratio = 0.2

# 路径（不要改）
img_train = "./dataset/images/train"
lab_train = "./dataset/labels/train"
img_val = "./dataset/images/val"
lab_val = "./dataset/labels/val"

# 创建文件夹
os.makedirs(img_val, exist_ok=True)
os.makedirs(lab_val, exist_ok=True)

# 获取所有图片
img_files = [f for f in os.listdir(img_train) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
random.shuffle(img_files)

# 计算要分割的数量
val_num = int(len(img_files) * split_ratio)
val_img_files = img_files[:val_num]

# 同时移动图片和对应标签
for img in val_img_files:
    # 移动图片
    shutil.move(os.path.join(img_train, img), os.path.join(img_val, img))

    # 构造标签名
    name = os.path.splitext(img)[0]
    lab = name + ".txt"

    # 移动标签
    lab_src = os.path.join(lab_train, lab)
    if os.path.exists(lab_src):
        shutil.move(lab_src, os.path.join(lab_val, lab))
    else:
        print(f"缺少标签: {lab}")

print(f"分割完成！共移动验证集：{len(img_files)} 张图片，{len(val_img_files)} 张作为验证集")
