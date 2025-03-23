import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO(model="yolov8x.pt")

# 视频文件
video_path = "nanwangjinxiao.mp4"
output_path = "output.mp4"

# 打开视频
cap = cv2.VideoCapture(video_path)

# 获取视频的宽、高、帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定义视频编码器和输出视频对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    # 获取图像
    res, frame = cap.read()
    # 如果读取成功
    if res:
        # 正向推理
        results = model(frame)

        # 绘制结果
        annotated_frame = results[0].plot()

        # 写入视频文件
        out.write(annotated_frame)

        # 显示图像（可选）
        cv2.imshow("YOLOV8", annotated_frame)

        # 按ESC退出
        if cv2.waitKey(1) == 27:
            break
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

print("视频保存完成：", output_path)
