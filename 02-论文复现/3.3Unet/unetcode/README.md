# UNet肺部分割

这个项目使用UNet模型对肺部X光图像进行语义分割，实现肺部区域的自动分割。本项目基于VOC格式的数据集进行训练和评估。

## 项目结构

```
.
├── VOCDatasets/             # VOC格式的数据集
│   ├── JPEGImages/          # 原始X光图像
│   ├── SegmentationClassPNG/# 分割标注(PNG格式)
│   ├── ImageSets/           # 数据集划分
│   ├── labels.txt           # 标签列表
│   └── colormap.txt         # 颜色映射
├── train_unet.py            # 训练UNet模型的脚本
├── predict_unet.py          # 用于单张图像预测的脚本
└── README.md                # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- Pillow
- matplotlib
- tqdm
- scikit-learn

安装依赖:

```bash
pip install torch torchvision numpy Pillow matplotlib tqdm scikit-learn
```

## 模型架构

本项目使用UNet架构进行语义分割。UNet由编码器和解码器组成:

- 编码器: 通过连续的卷积和池化层提取图像特征
- 解码器: 通过上采样和卷积恢复空间分辨率
- 跳跃连接: 连接编码器和解码器的对应层，保留空间信息

## 数据集

本项目使用VOC格式的数据集，包含以下类别:
- 背景
- 肺部

## 训练模型

运行以下命令开始训练UNet模型:

```bash
python train_unet.py
```

训练过程中会自动保存性能最佳的模型，并生成训练曲线图。

### 训练参数

您可以在`train_unet.py`文件中修改以下超参数:

- `BATCH_SIZE`: 批次大小
- `EPOCHS`: 训练轮数
- `LEARNING_RATE`: 学习率
- `IMAGE_HEIGHT`和`IMAGE_WIDTH`: 输入图像的高度和宽度

## 模型预测

训练完成后，可以使用`predict_unet.py`脚本对单张图像进行预测:

```bash
python predict_unet.py --image path/to/your/image.jpg --model best_unet_model.pth --output prediction_result.png
```

参数说明:
- `--image`: 输入图像路径（必需）
- `--model`: 模型权重文件路径，默认为'best_unet_model.pth'
- `--output`: 输出预测结果图像的保存路径，默认为'prediction_result.png'

## 评估指标

本项目使用以下指标评估分割性能:

- **交并比(IoU)**: 预测区域与真实区域的交集除以并集
- **Dice系数**: 预测与真实区域的相似度度量

## 结果可视化

训练过程中会生成:
- 训练和验证损失曲线
- 训练和验证IoU曲线
- 样本预测可视化结果

## 注意事项

- 建议在GPU上运行训练过程以加快速度
- 对于大规模数据集，可能需要调整批次大小以适应GPU内存
- 数据预处理中使用了ImageNet的均值和标准差进行归一化

## 参考

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html) 