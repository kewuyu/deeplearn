#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import argparse
from PIL import Image, ImageDraw
import shutil
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='将labelme格式转换为VOC格式')
    parser.add_argument('--input_dir', default='./datasets', help='输入labelme格式数据目录')
    parser.add_argument('--output_dir', default='./VOCDatasets', help='输出VOC格式数据目录')
    parser.add_argument('--img_dir', default=None, help='自定义图像文件夹路径，如果不指定则使用input_dir/img')
    parser.add_argument('--json_dir', default=None, help='自定义标注JSON文件夹路径，如果不指定则使用input_dir/labels')
    parser.add_argument('--labels_file', default=None, help='自定义标签文件路径，如果不指定则使用input_dir/labels.txt')
    parser.add_argument('--img_format', default='jpg', choices=['jpg', 'png'], help='保存图像的格式，可选jpg或png')
    args = parser.parse_args()
    return args

def process_label_name(label_file):
    # 读取标签名称
    with open(label_file, 'r') as f:
        labels = f.read().splitlines()
    # 过滤掉 __ignore__ 和 _background_
    labels = [label for label in labels if not (label.startswith('__') or label.startswith('_'))]
    return {i+1: label for i, label in enumerate(labels)}

def create_voc_directories(output_dir):
    # 创建VOC标准目录结构
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'SegmentationClassPNG'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'ImageSets', 'Segmentation'), exist_ok=True)

def convert_labelme_to_voc(img_dir, label_dir, output_dir, label_map, img_format='jpg'):
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    file_names = []
    
    print(f"处理 {len(json_files)} 个文件...")
    for json_file in tqdm(json_files):
        file_name = json_file.split('.')[0]
        file_names.append(file_name)
        
        # 读取JSON文件
        with open(os.path.join(label_dir, json_file), 'r') as f:
            data = json.load(f)
        
        # 获取图像路径和尺寸
        # 根据JSON文件中的相对路径或直接使用文件名查找图像
        img_filename = os.path.basename(data['imagePath'])
        img_path = os.path.join(img_dir, img_filename)
        
        # 如果图像不存在，尝试直接使用JSON文件名对应的图像名
        if not os.path.exists(img_path):
            possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
            for ext in possible_extensions:
                alt_img_path = os.path.join(img_dir, file_name + ext)
                if os.path.exists(alt_img_path):
                    img_path = alt_img_path
                    break
        
        if not os.path.exists(img_path):
            print(f"警告: 找不到图像 {img_filename}，跳过此文件")
            continue
        
        img_height = data['imageHeight']
        img_width = data['imageWidth']
        
        # 创建空白掩码图像（单通道，黑色背景）
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # 处理每个形状
        for shape in data['shapes']:
            label = shape['label']
            if label in [l for l in label_map.values()]:
                # 找到标签索引
                label_index = [k for k, v in label_map.items() if v == label][0]
                
                # 获取多边形点
                points = np.array(shape['points'], dtype=np.int32)
                
                # 在掩码上绘制多边形
                if shape['shape_type'] == 'polygon':
                    cv2.fillPoly(mask, [points], label_index)
        
        # 保存图像
        try:
            img = Image.open(img_path)
            
            # 确保图像是RGB模式，这样可以保存为JPEG
            if img_format.lower() == 'jpg' and img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 根据指定的格式保存图像
            if img_format.lower() == 'jpg':
                img.save(os.path.join(output_dir, 'JPEGImages', f"{file_name}.jpg"), quality=95)
            else:
                img.save(os.path.join(output_dir, 'JPEGImages', f"{file_name}.png"))
            
        except Exception as e:
            print(f"警告: 保存图像 {file_name} 时出错: {e}")
            continue
        
        # 保存掩码图像（单通道PNG）
        mask_pil = Image.fromarray(mask)
        mask_pil.save(os.path.join(output_dir, 'SegmentationClassPNG', f"{file_name}.png"))
        
        # 保存颜色编码的掩码（用于可视化）
        color_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        for label_id in label_map.keys():
            color_mask[mask == label_id] = [label_id * 10, label_id * 20, label_id * 30]  # 简单的颜色映射
        
        color_mask_pil = Image.fromarray(color_mask)
        color_mask_pil.save(os.path.join(output_dir, 'SegmentationClass', f"{file_name}.png"))
    
    # 创建训练和验证集
    train_val_split = int(len(file_names) * 0.8)
    train_files = file_names[:train_val_split]
    val_files = file_names[train_val_split:]
    
    # 保存训练和验证集文件
    with open(os.path.join(output_dir, 'ImageSets', 'Segmentation', 'train.txt'), 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(os.path.join(output_dir, 'ImageSets', 'Segmentation', 'val.txt'), 'w') as f:
        f.write('\n'.join(val_files))
    
    with open(os.path.join(output_dir, 'ImageSets', 'Segmentation', 'trainval.txt'), 'w') as f:
        f.write('\n'.join(file_names))

def create_colormap_file(output_dir, label_map):
    # 创建标签颜色映射文件
    with open(os.path.join(output_dir, 'colormap.txt'), 'w') as f:
        f.write('0 0 0 0 background\n')  # 背景色
        for label_id, label_name in label_map.items():
            # 简单的颜色映射
            r = (label_id * 10) % 255
            g = (label_id * 20) % 255
            b = (label_id * 30) % 255
            f.write(f'{label_id} {r} {g} {b} {label_name}\n')

def main():
    args = parse_args()
    
    # 创建VOC格式目录
    create_voc_directories(args.output_dir)
    
    # 设置图像和标注JSON文件夹路径
    img_dir = args.img_dir if args.img_dir else os.path.join(args.input_dir, 'img')
    json_dir = args.json_dir if args.json_dir else os.path.join(args.input_dir, 'labels')
    labels_file = args.labels_file if args.labels_file else os.path.join(args.input_dir, 'labels.txt')
    
    # 检查目录是否存在
    if not os.path.exists(img_dir):
        print(f"错误: 图像目录 {img_dir} 不存在!")
        return
    
    if not os.path.exists(json_dir):
        print(f"错误: 标注JSON目录 {json_dir} 不存在!")
        return
    
    if not os.path.exists(labels_file):
        print(f"错误: 标签文件 {labels_file} 不存在!")
        return
    
    # 处理标签名称
    label_map = process_label_name(labels_file)
    
    # 打印处理信息
    print(f"图像目录: {img_dir}")
    print(f"标注JSON目录: {json_dir}")
    print(f"标签文件: {labels_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"图像保存格式: {args.img_format}")
    print(f"找到的标签: {', '.join(label_map.values())}")
    
    # 转换数据
    convert_labelme_to_voc(img_dir, json_dir, args.output_dir, label_map, args.img_format)
    
    # 创建颜色映射文件
    create_colormap_file(args.output_dir, label_map)
    
    # 保存标签映射
    with open(os.path.join(args.output_dir, 'labels.txt'), 'w') as f:
        f.write('background\n')  # 添加背景类
        for _, label_name in label_map.items():
            f.write(f'{label_name}\n')
    
    print(f"转换完成！数据已保存到 {args.output_dir}")

if __name__ == '__main__':
    main() 