#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

def split_yolo_dataset(dataset_dir, output_dir, train_ratio=0.8, seed=42):
    """
    将YOLO格式的数据集划分为训练集和验证集
    
    参数:
        dataset_dir (str): 原始数据集目录，应包含images和labels子目录
        output_dir (str): 输出目录
        train_ratio (float): 训练集比例，默认0.8
        seed (int): 随机种子，确保可重复性，默认42
    """
    random.seed(seed)
    
    # 创建输出目录结构
    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    
    train_img_dir = os.path.join(output_img_dir, 'train')
    train_label_dir = os.path.join(output_label_dir, 'train')
    val_img_dir = os.path.join(output_img_dir, 'val')
    val_label_dir = os.path.join(output_label_dir, 'val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # 确定输入目录
    img_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')
    
    if not os.path.exists(img_dir) or not os.path.exists(label_dir):
        print(f"错误: 未找到图像或标签目录:\n{img_dir}\n{label_dir}")
        return
    
    # 获取所有图像文件
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = []
    
    for ext in img_extensions:
        img_files.extend(list(Path(img_dir).glob(f'*{ext}')))
    
    # 确保文件名排序，便于后续调试
    img_files.sort()
    total_files = len(img_files)
    
    if total_files == 0:
        print("错误: 未找到有效的图像文件")
        return
    
    print(f"找到 {total_files} 个图像文件")
    
    # 随机打乱文件列表
    random.shuffle(img_files)
    
    # 计算训练集大小
    train_size = int(total_files * train_ratio)
    
    # 划分数据集
    train_files = img_files[:train_size]
    val_files = img_files[train_size:]
    
    print(f"划分为 {len(train_files)} 个训练样本和 {len(val_files)} 个验证样本")
    
    # 复制训练集文件
    print("正在复制训练集文件...")
    for img_path in tqdm(train_files):
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        
        # 图像文件路径
        src_img = str(img_path)
        dst_img = os.path.join(train_img_dir, img_filename)
        
        # 标签文件路径
        src_label = os.path.join(label_dir, label_filename)
        dst_label = os.path.join(train_label_dir, label_filename)
        
        # 复制图像文件
        shutil.copy2(src_img, dst_img)
        
        # 如果存在标签文件，则复制
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    # 复制验证集文件
    print("正在复制验证集文件...")
    for img_path in tqdm(val_files):
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        
        # 图像文件路径
        src_img = str(img_path)
        dst_img = os.path.join(val_img_dir, img_filename)
        
        # 标签文件路径
        src_label = os.path.join(label_dir, label_filename)
        dst_label = os.path.join(val_label_dir, label_filename)
        
        # 复制图像文件
        shutil.copy2(src_img, dst_img)
        
        # 如果存在标签文件，则复制
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)
    
    # 生成数据集配置文件
    dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w', encoding='utf-8') as f:
        f.write(f"# YOLOv5/YOLOv8 数据集配置\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: 0  # 类别数量，请根据实际情况修改\n")
        f.write(f"names: []  # 类别名称，请根据实际情况修改\n")
    
    print(f"数据集划分完成!")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"配置文件: {dataset_yaml}")
    print(f"注意：请编辑 {dataset_yaml} 文件，设置正确的类别数量和类别名称")

def count_classes(label_dir):
    """统计标签目录中的类别数量"""
    class_ids = set()
    for label_file in Path(label_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    return class_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将YOLO格式数据集划分为训练集和验证集")
    parser.add_argument("dataset_dir", help="原始数据集目录，应包含images和labels子目录")
    parser.add_argument("--output_dir", default="datasets", help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例，默认0.8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保可重复性，默认42")
    
    args = parser.parse_args()
    
    split_yolo_dataset(args.dataset_dir, args.output_dir, args.train_ratio, args.seed) 