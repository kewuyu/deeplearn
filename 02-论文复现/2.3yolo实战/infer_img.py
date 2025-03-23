import cv2
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8x.pt")  # 确保路径正确

# 读取图像
image_path = "datasets/images/val/frame_000003.jpg"
image = cv2.imread(image_path)

# 检查图像是否成功加载
if image is None:
    print(f"错误：无法加载图像 {image_path}，请检查路径！")
    exit()

# 进行目标检测
results = model(image)

# 绘制检测结果
annotated_image = results[0].plot()

# 保存推理结果
output_path = "output_image.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"推理完成，结果已保存至 {output_path}")
