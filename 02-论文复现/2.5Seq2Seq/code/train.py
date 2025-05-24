"""
Seq2Seq机器翻译模型训练脚本
包含训练循环、验证、模型保存、翻译测试等功能
支持GPU/CPU训练，自动保存最佳模型和训练曲线
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import Config
from data_loader import prepare_data, get_data_loaders
from models import create_model, count_parameters
from vocabulary import Vocabulary

def train_epoch(model, train_loader, optimizer, criterion, clip):
    """
    训练一个epoch
    
    Args:
        model (Seq2Seq): 待训练的模型
        train_loader (DataLoader): 训练数据加载器
        optimizer (Optimizer): 优化器
        criterion (Loss): 损失函数
        clip (float): 梯度裁剪阈值
        
    Returns:
        float: 平均训练损失
    """
    model.train()  # 设置为训练模式
    epoch_loss = 0
    
    # 遍历训练批次
    for i, (src, trg) in enumerate(tqdm(train_loader, desc="Training")):
        # 将数据移动到指定设备（GPU/CPU）
        src, trg = src.to(Config.DEVICE), trg.to(Config.DEVICE)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播：使用Teacher Forcing
        output = model(src, trg)
        
        # 计算损失（忽略第一个SOS标记）
        output_dim = output.shape[-1]
        # 重塑输出和目标张量用于损失计算
        output = output[:, 1:].contiguous().view(-1, output_dim)  # (batch_size * (seq_len-1), vocab_size)
        trg = trg[:, 1:].contiguous().view(-1)  # (batch_size * (seq_len-1),)
        
        # 计算交叉熵损失
        loss = criterion(output, trg)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # 更新参数
        optimizer.step()
        
        # 累积损失
        epoch_loss += loss.item()
        
        # 定期打印训练信息
        if (i + 1) % Config.PRINT_EVERY == 0:
            print(f'Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # 返回平均损失
    return epoch_loss / len(train_loader)

def evaluate(model, val_loader, criterion):
    """
    在验证集上评估模型
    
    Args:
        model (Seq2Seq): 待评估的模型
        val_loader (DataLoader): 验证数据加载器
        criterion (Loss): 损失函数
        
    Returns:
        float: 平均验证损失
    """
    model.eval()  # 设置为评估模式
    epoch_loss = 0
    
    # 禁用梯度计算以节省内存和计算
    with torch.no_grad():
        for src, trg in tqdm(val_loader, desc="Evaluating"):
            src, trg = src.to(Config.DEVICE), trg.to(Config.DEVICE)
            
            # 关闭Teacher Forcing进行验证（更接近实际推理）
            output = model(src, trg, teacher_forcing_ratio=0)
            
            # 计算损失
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(val_loader)

def epoch_time(start_time, end_time):
    """
    计算并格式化训练时间
    
    Args:
        start_time (float): 开始时间
        end_time (float): 结束时间
        
    Returns:
        tuple: (分钟, 秒)
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def translate_sentence(model, sentence, input_vocab, output_vocab, max_length=50):
    """
    翻译单个句子（用于测试）
    
    Args:
        model (Seq2Seq): 训练好的模型
        sentence (str): 待翻译的英文句子
        input_vocab (Vocabulary): 输入语言词汇表
        output_vocab (Vocabulary): 输出语言词汇表
        max_length (int): 生成序列的最大长度
        
    Returns:
        str: 翻译结果
    """
    model.eval()
    
    with torch.no_grad():
        # 预处理输入句子
        tokens = input_vocab.sentence_to_indexes(sentence)
        # 添加SOS和EOS标记
        tokens = [Config.SOS_TOKEN] + tokens + [Config.EOS_TOKEN]
        
        # 转换为张量并添加批次维度
        src_tensor = torch.tensor(tokens).unsqueeze(0).to(Config.DEVICE)
        
        # 使用模型翻译
        trg_indexes = model.translate(src_tensor, max_length)
        
        # 将索引序列转换回文本
        trg_tokens = trg_indexes[0].tolist()
        return output_vocab.indexes_to_sentence(trg_tokens)

def test_translation(model, input_vocab, output_vocab):
    """
    测试模型翻译效果
    
    Args:
        model (Seq2Seq): 待测试的模型
        input_vocab (Vocabulary): 输入语言词汇表
        output_vocab (Vocabulary): 输出语言词汇表
    """
    # 测试句子集合
    test_sentences = [
        "Hello!",
        "Good morning.",
        "How are you?",
        "I love you.",
        "Thank you very much."
    ]
    
    print("\n=== 翻译测试 ===")
    for sentence in test_sentences:
        translation = translate_sentence(model, sentence, input_vocab, output_vocab)
        print(f"EN: {sentence}")
        print(f"ZH: {translation}")
        print("-" * 50)

def plot_losses(train_losses, val_losses):
    """
    绘制并保存训练损失曲线
    
    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

def main():
    """
    主训练函数，协调整个训练流程
    """
    print("=" * 60)
    print("Seq2Seq机器翻译模型训练")
    print("=" * 60)
    
    # 1. 准备数据
    print("准备数据...")
    train_pairs, val_pairs, test_pairs, input_vocab, output_vocab = prepare_data()
    
    # 2. 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader = get_data_loaders(
        train_pairs, val_pairs, test_pairs, input_vocab, output_vocab
    )
    
    # 3. 创建模型
    print("创建模型...")
    model = create_model(input_vocab.n_words, output_vocab.n_words)
    
    print(f'模型参数数量: {count_parameters(model):,}')
    print(f'训练设备: {Config.DEVICE}')
    
    # 4. 保存词汇表
    print("保存词汇表...")
    input_vocab.save(f'input_{Config.VOCAB_SAVE_PATH}')
    output_vocab.save(f'output_{Config.VOCAB_SAVE_PATH}')
    
    # 5. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # 使用交叉熵损失，忽略填充标记
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_TOKEN)
    
    # 6. 训练历史记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    # 7. 训练循环
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip=1)
        
        # 在验证集上评估
        val_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 计算训练时间
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # 打印训练信息
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'best_{Config.MODEL_SAVE_PATH}')
            print(f'\t新的最佳模型已保存! Val Loss: {val_loss:.3f}')
        
        # 定期保存模型检查点
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'epoch_{epoch+1}_{Config.MODEL_SAVE_PATH}')
        
        # 定期测试翻译效果
        if (epoch + 1) % 10 == 0:
            test_translation(model, input_vocab, output_vocab)
        
        print()
    
    # 8. 保存最终模型
    print("保存最终模型...")
    torch.save({
        'epoch': Config.NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, f'final_{Config.MODEL_SAVE_PATH}')
    
    # 9. 绘制损失曲线
    print("绘制训练曲线...")
    plot_losses(train_losses, val_losses)
    
    # 10. 最终测试
    print("\n=== 最终翻译测试 ===")
    test_translation(model, input_vocab, output_vocab)
    
    # 11. 在测试集上评估
    print("在测试集上评估...")
    test_loss = evaluate(model, test_loader, criterion)
    print(f'\nTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}')
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

if __name__ == '__main__':
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始训练
    main() 