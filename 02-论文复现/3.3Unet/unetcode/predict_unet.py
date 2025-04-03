import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

# 导入模型定义
from train_unet import UNet

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
NUM_CLASSES = 2  # 背景 + 肺部
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

def predict_single_image(model, image_path, save_path=None):
    """
    在单张图像上进行预测并可视化结果
    
    Args:
        model: 训练好的UNet模型
        image_path: 输入图像路径
        save_path: 保存结果的路径，如果为None则不保存
    """
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap='gray')
    plt.title('预测结果 (白色: 肺部)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"结果已保存至 {save_path}")
    
    plt.show()
    
    # 返回预测结果
    return pred

def main():
    parser = argparse.ArgumentParser(description='UNet Lung Segmentation Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='best_unet_model.pth', help='Path to the model weights')
    parser.add_argument('--output', type=str, default='prediction_result.png', help='Path to save the output image')
    
    args = parser.parse_args()
    
    # 加载模型
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"模型已从 {args.model} 加载")
    
    # 进行预测
    predict_single_image(model, args.image, args.output)

if __name__ == '__main__':
    main() 