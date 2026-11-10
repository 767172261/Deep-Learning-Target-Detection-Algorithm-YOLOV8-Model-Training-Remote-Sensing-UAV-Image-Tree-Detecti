深度学习目标检测算法YOLOV8模型训练 遥感无人机图像树木检测数据集 无人机树木检测数据集
@[toc]


遥感无人机图像树木检测数据集 yolo和voc两种标注方式

标签数：8158
框数: 327121

1个类别以及框数：
Tree (327121)


### **遥感图像树木检测数据集统计**

| **项目** | **数量/类型** |
| :--- | :--- |
| **数据集格式** | YOLO, VOC |
| **总标签数** | 8,158 |
| **总边界框数** | 327,121 |
| **类别数量** | 1 |
| **类别名称** | Tree (树木) |

---

### **按类别统计**

| **类别** | **边界框数量** | **占比** |
| :--- | :--- | :--- |
| Tree (树木) | 327,121 | 100% |
| **总计** | **327,121** | **100%** |

> **说明**：该数据集仅包含“Tree”一个类别，所有边界框均用于标注树木。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/409315f6f2794fc8a8ee07b51c017d6c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff53c93116ac4d2aafaa8b498b56093b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a0b7c622fe7a4ddfa82311705944d5d4.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4cb9a5c8250647a5aa66f4990079dc15.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/42ebea36389341fd87fd881098a6458b.png)
1


---

## 🌳 遥感图像树木检测数据集 (DST1703) 概览

| 项目 | 说明 |
| :--- | :--- |
| **数据集名称** | DST1703 |
| **任务类型** | 单类别目标检测（Tree Detection） |
| **图像总数** | 8,158 张 |
| **标注框总数** | 327,121 个（平均每图约 40 个树） |
| **类别数量** | 1 类：`Tree` |
| **标注格式** | 同时提供 **YOLO `.txt`** 和 **Pascal VOC `.xml`** 格式 |
| **图像分辨率** | 高分辨率遥感图（常见 1024×1024 或 2048×2048） |
| **适用模型** | YOLOv8n / YOLOv8s（轻量高效，适合密集小目标） |
| **挑战点** | 树木密集、尺度变化大、阴影干扰、背景复杂 |

---

## 📂 博主推荐目录结构（YOLO 格式）

```bash
dst1703_dataset/
│
├── images/
│   ├── train/          # 训练图像
│   ├── val/            # 验证图像
│   └── test/           # 测试图像
│
├── labels/
│   ├── train/          # YOLO 格式 .txt 文件
│   ├── val/
│   └── test/
│
├── annotations/        # 可选：VOC XML 文件
│
├── data.yaml           # YOLO 配置文件
└── README.md
```

> ✅ 建议按 **8:1:1** 划分训练/验证/测试集。

---

## 🔄 VOC XML 转 YOLO TXT 脚本（如需）

```python
# voc_to_yolo_tree.py
import os
import xml.etree.ElementTree as ET
from pathlib import Path

# ==================== 配置 ====================
class_name = 'tree'
class_to_id = {class_name: 0}  # 单类别，ID=0
voc_dir = "dst1703_dataset/annotations"      # VOC XML 所在目录
yolo_label_dir = "dst1703_dataset/labels/train"  # 输出目录
img_width = 1024   # 假设图像尺寸，建议根据实际图像读取
img_height = 1024

os.makedirs(yolo_label_dir, exist_ok=True)

def convert_xml_to_yolo(xml_file, width=img_width, height=img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text.lower().strip()
        if cls_name != 'tree':
            continue  # 只保留 tree 类

        cls_id = class_to_id['tree']
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 归一化
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # 保存 .txt
    txt_file = Path(yolo_label_dir) / (Path(xml_file).stem + ".txt")
    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_lines))

# 批量转换
for xml_file in Path(voc_dir).glob("*.xml"):
    convert_xml_to_yolo(xml_file)

print("✅ VOC → YOLO 转换完成！")
```

---

## 📄 `data.yaml` 配置文件

```yaml
# data.yaml
train: ./dst1703_dataset/images/train
val: ./dst1703_dataset/images/val
test: ./dst1703_dataset/images/test

# 类别数量
nc: 1

# 类别名称
names: ['tree']
```

---

## 🚀 YOLOv8 详细训练代码（单类别树木检测）

```python
# train_tree.py
from ultralytics import YOLO
import torch
import os

# ==================== 配置参数 ====================
MODEL_TYPE = 'yolov8s.pt'           # 推荐：yolov8n.pt（更快），yolov8m.pt（更高精度）
DATA_CONFIG = 'dst1703_dataset/data.yaml'
EPOCHS = 150
BATCH_SIZE = 32                     # 根据 GPU 显存调整（16/32/64）
IMG_SIZE = 1024                     # 遥感图建议使用 1024 或 1280
PROJECT_NAME = 'tree_detection_dst1703'
EXPERIMENT_NAME = 'yolov8s_1024_tree_dst1703'

# 创建项目目录
os.makedirs(f"runs/train/{EXPERIMENT_NAME}", exist_ok=True)

# ==================== 设备检查 ====================
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")

# ==================== 加载模型 ====================
model = YOLO(MODEL_TYPE)

# ==================== 开始训练 ====================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,
    save_period=10,
    device=DEVICE,
    workers=4,
    optimizer='AdamW',
    lr0=1e-3,
    lrf=1e-2,
    momentum=0.937,
    weight_decay=1e-4,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    close_mosaic=10,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,           # 更强旋转增强（遥感图方向多变）
    translate=0.3,
    scale=0.7,              # 更大尺度变化
    shear=3.0,
    flipud=0.5,             # 上下翻转（遥感图对称）
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.2,         # 对密集小树非常有效
    patience=40,            # 早停
)

print("🎉 训练完成！")
print(f"📌 模型保存路径: runs/train/{EXPERIMENT_NAME}")
print(f"📈 最佳 mAP@0.5: {results.best_box_map:.4f}")
```

---

## 🔍 模型验证与评估

```python
# validate.py
from ultralytics import YOLO

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')
metrics = model.val(
    data=DATA_CONFIG,
    split='val',
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    save_json=True,
    conf=0.001,
    iou=0.5,
)

print(f"📊 mAP@0.5: {metrics.box.map:.4f}")
print(f"📊 mAP@0.5:0.95: {metrics.box.map50:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")
```

---

## 🔦 推理检测代码（支持图片、视频、大图切片）

```python
# predict.py
from ultralytics import YOLO

# ==================== 配置 ====================
MODEL_PATH = f'runs/train/{EXPERIMENT_NAME}/weights/best.pt'
SOURCE = 'dst1703_dataset/images/test'  # 可为：图片、目录、视频、摄像头
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
SHOW = False
SAVE = True
PROJECT = 'runs/predict/tree_dst1703'

# ==================== 加载模型 ====================
model = YOLO(MODEL_PATH)

# ==================== 推理 ====================
results = model.predict(
    source=SOURCE,
    imgsz=IMG_SIZE,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    device=DEVICE,
    show=SHOW,
    save=SAVE,
    project=PROJECT,
    name='result',
    exist_ok=True,
    classes=None,
    line_width=1,
    verbose=True,
    max_det=1000,  # 单图最多检测数（树木密集）
)

# 打印结果
for r in results:
    print(f"🖼️ {r.path}: {len(r.boxes)} 棵树被检测到")
```

---

## 🖼️ 大尺寸遥感图切片推理（推荐）

由于遥感图通常很大（如 4000×4000），建议先切片再检测：

```python
# slice_and_detect.py
from ultralytics import YOLO
import cv2
import numpy as np

def slice_image(image_path, slice_size=1024, overlap=200):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    slices = []
    coords = []

    for y in range(0, h, slice_size - overlap):
        for x in range(0, w, slice_size - overlap):
            x_end = min(x + slice_size, w)
            y_end = min(y + slice_size, h)
            slice_img = img[y:y_end, x:x_end]
            slices.append(slice_img)
            coords.append((x, y, x_end, y_end))
    return slices, coords, img.shape

# 加载模型
model = YOLO('runs/train/yolov8s_1024_tree_dst1703/weights/best.pt')
slices, coords, orig_shape = slice_image('large_satellite_image.jpg')

all_boxes = []
for i, slice_img in enumerate(slices):
    results = model(slice_img, conf=0.3, imgsz=1024)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    # 转换回原图坐标
    x_off, y_off = coords[i][0], coords[i][1]
    boxes[:, [0, 2]] += x_off
    boxes[:, [1, 3]] += y_off
    all_boxes.extend(boxes)

print(f"✅ 共检测到 {len(all_boxes)} 棵树")
```

---

## ✅ 训练优化建议（遥感图专用）

| 技巧 | 说明 |
|------|------|
| **Copy-Paste Augmentation** | 将树粘贴到新背景，提升小树/遮挡树检出率 |
| **高分辨率训练（1024/1280）** | 保留更多细节，适合小目标 |
| **TTA（Test-Time Augmentation）** | 推理时多尺度+翻转提升召回 |
| **Mosaic 增强强度调高** | `mosaic=1.0` + `mixup=0.2` |
| **使用 YOLOv8m/l** | 若精度不足，可升级模型 |

---

## 📈 预期性能（YOLOv8s, 1024×1024）

| 指标 | 预期值 |
|------|--------|
| mAP@0.5 | 0.88 – 0.94 |
| 推理速度 | 50 – 80 FPS（Tesla T4） |
| 模型大小 | ~29 MB |

---

遥感树木检测数据集，配合 YOLOv8 可快速构建：

- 城市绿化地图生成系统
- 森林资源自动清点工具
- 生态环境监测平台
- 智慧林业 AI 引擎


