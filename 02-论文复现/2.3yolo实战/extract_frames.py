#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, prefix='frame', fps_extract=None):
    """
    从视频中提取帧并保存为图像文件
    
    参数:
        video_path (str): 输入视频的路径
        output_dir (str): 输出帧的目录
        prefix (str): 输出文件名前缀
        fps_extract (float): 提取的帧率，如果为None则提取所有帧，如果为1则每秒提取一帧
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的一些基本信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"视频信息:")
    print(f"- 总帧数: {frame_count}")
    print(f"- 原始FPS: {original_fps}")
    
    # 计算需要跳过的帧数
    if fps_extract is not None:
        frames_to_skip = int(original_fps / fps_extract) - 1
        expected_frames = int(frame_count / (frames_to_skip + 1)) + 1
        print(f"- 提取FPS: {fps_extract}")
        print(f"- 估计将提取: ~{expected_frames} 帧")
    else:
        frames_to_skip = 0
        expected_frames = frame_count
        print(f"- 提取: 所有帧")
    
    # 提取帧
    frame_idx = 0
    saved_idx = 0
    skip_counter = 0
    
    with tqdm(total=expected_frames, desc="提取帧") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 根据设定的提取帧率决定是否保存当前帧
            if fps_extract is None or skip_counter == 0:
                # 保存帧为图像文件
                output_path = os.path.join(output_dir, f"{prefix}_{saved_idx:06d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_idx += 1
                pbar.update(1)
            
            # 更新跳帧计数器
            if frames_to_skip > 0:
                skip_counter = (skip_counter + 1) % (frames_to_skip + 1)
            
            frame_idx += 1
    
    # 释放资源
    cap.release()
    print(f"已完成! 共提取了 {saved_idx} 帧图像到 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中提取帧")
    parser.add_argument("video_path", help="输入视频的路径")
    parser.add_argument("--output_dir", default="frames", help="保存帧的输出目录")
    parser.add_argument("--prefix", default="frame", help="输出文件名前缀")
    parser.add_argument("--fps", type=float, default=1, help="提取的帧率，默认为1fps（每秒1帧），设置为0表示提取所有帧")
    
    args = parser.parse_args()
    
    # 如果fps设置为0，则提取所有帧
    fps_extract = None if args.fps == 0 else args.fps
    
    extract_frames(args.video_path, args.output_dir, args.prefix, fps_extract) 