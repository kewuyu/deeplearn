"""
数据加载和预处理模块
负责读取CMN数据集、数据预处理、构建词汇表、创建数据加载器等功能
"""

import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from vocabulary import Vocabulary
from config import Config

class TranslationDataset(Dataset):
    """
    翻译数据集类，用于PyTorch DataLoader
    将句子对转换为索引序列，支持批量处理
    """
    
    def __init__(self, pairs, input_vocab, output_vocab):
        """
        初始化数据集
        
        Args:
            pairs (list): 句子对列表，每个元素为[英文句子, 中文句子]
            input_vocab (Vocabulary): 输入语言（英文）词汇表
            output_vocab (Vocabulary): 输出语言（中文）词汇表
        """
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (输入序列张量, 输出序列张量)
        """
        input_sentence, output_sentence = self.pairs[idx]
        
        # 将句子转换为索引序列
        input_indexes = self.input_vocab.sentence_to_indexes(input_sentence)
        output_indexes = self.output_vocab.sentence_to_indexes(output_sentence)
        
        # 在序列末尾添加EOS标记
        input_indexes.append(Config.EOS_TOKEN)
        output_indexes.append(Config.EOS_TOKEN)
        
        # 转换为PyTorch张量
        return torch.tensor(input_indexes, dtype=torch.long), torch.tensor(output_indexes, dtype=torch.long)

def collate_fn(batch):
    """
    批次整理函数，用于DataLoader
    将不同长度的序列填充到相同长度
    
    Args:
        batch (list): 批次数据，包含多个(input_seq, target_seq)元组
        
    Returns:
        tuple: (填充后的输入序列, 填充后的目标序列)
    """
    # 分离输入序列和目标序列
    input_seqs, target_seqs = zip(*batch)
    
    # 使用PAD_TOKEN填充序列到相同长度
    input_seqs = pad_sequence(input_seqs, batch_first=True, padding_value=Config.PAD_TOKEN)
    target_seqs = pad_sequence(target_seqs, batch_first=True, padding_value=Config.PAD_TOKEN)
    
    return input_seqs, target_seqs

def read_data(filename):
    """
    读取CMN数据文件
    
    Args:
        filename (str): 数据文件路径
        
    Returns:
        list: 句子对列表，每个元素为[英文, 中文]
    """
    print(f"读取数据文件: {filename}")
    
    # 读取文件内容并按行分割
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    
    pairs = []
    for line in lines:
        # CMN数据格式：英文\t中文
        if '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 2:
                english = parts[0].strip()
                chinese = parts[1].strip()
                pairs.append([english, chinese])
    
    print(f"读取到 {len(pairs)} 个句子对")
    return pairs

def filter_pairs(pairs, max_length):
    """
    过滤句子对，去除过长的句子
    
    Args:
        pairs (list): 原始句子对列表
        max_length (int): 句子最大长度
        
    Returns:
        list: 过滤后的句子对列表
    """
    filtered_pairs = []
    for pair in pairs:
        # 检查英文句子词数和中文句子字符数
        english_len = len(pair[0].split())
        chinese_len = len(list(pair[1]))
        
        if english_len < max_length and chinese_len < max_length:
            filtered_pairs.append(pair)
    
    print(f"过滤后剩余 {len(filtered_pairs)} 个句子对")
    return filtered_pairs

def prepare_data():
    """
    准备训练数据的主函数
    读取数据、构建词汇表、分割数据集
    
    Returns:
        tuple: (训练集, 验证集, 测试集, 输入词汇表, 输出词汇表)
    """
    # 1. 读取原始数据
    pairs = read_data(Config.DATA_PATH)
    
    # 2. 过滤过长的句子
    pairs = filter_pairs(pairs, Config.MAX_LENGTH)
    
    # 3. 随机打乱数据
    random.shuffle(pairs)
    
    # 4. 创建词汇表
    input_vocab = Vocabulary('english')    # 英文词汇表
    output_vocab = Vocabulary('chinese')   # 中文词汇表
    
    print("构建词汇表...")
    # 遍历所有句子对，构建词汇表
    for pair in pairs:
        input_vocab.add_sentence(pair[0])   # 添加英文句子
        output_vocab.add_sentence(pair[1])  # 添加中文句子
    
    # 5. 修剪低频词汇
    input_vocab.trim_vocab(Config.MIN_COUNT)
    output_vocab.trim_vocab(Config.MIN_COUNT)
    
    print(f"英文词汇表大小: {input_vocab.n_words}")
    print(f"中文词汇表大小: {output_vocab.n_words}")
    
    # 6. 按比例分割数据集
    total_size = len(pairs)
    train_size = int(total_size * Config.TRAIN_RATIO)
    val_size = int(total_size * Config.VAL_RATIO)
    
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]
    
    print(f"训练集: {len(train_pairs)}")
    print(f"验证集: {len(val_pairs)}")
    print(f"测试集: {len(test_pairs)}")
    
    return train_pairs, val_pairs, test_pairs, input_vocab, output_vocab

def get_data_loaders(train_pairs, val_pairs, test_pairs, input_vocab, output_vocab):
    """
    创建PyTorch数据加载器
    
    Args:
        train_pairs (list): 训练句子对
        val_pairs (list): 验证句子对
        test_pairs (list): 测试句子对
        input_vocab (Vocabulary): 输入词汇表
        output_vocab (Vocabulary): 输出词汇表
        
    Returns:
        tuple: (训练加载器, 验证加载器, 测试加载器)
    """
    # 创建数据集对象
    train_dataset = TranslationDataset(train_pairs, input_vocab, output_vocab)
    val_dataset = TranslationDataset(val_pairs, input_vocab, output_vocab)
    test_dataset = TranslationDataset(test_pairs, input_vocab, output_vocab)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,                    # 训练时打乱数据
        collate_fn=collate_fn           # 使用自定义整理函数
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,                   # 验证时不打乱
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,                   # 测试时不打乱
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader 