"""
Seq2Seq模型定义模块
包含编码器、解码器和完整的Seq2Seq模型实现
使用LSTM架构，支持双向编码器和Teacher Forcing训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from config import Config

class Encoder(nn.Module):
    """
    编码器类，使用LSTM将输入序列编码为固定长度的上下文向量
    支持双向LSTM和多层结构
    """
    
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        """
        初始化编码器
        
        Args:
            input_size (int): 输入词汇表大小
            embedding_size (int): 词嵌入维度
            hidden_size (int): LSTM隐藏层大小
            num_layers (int): LSTM层数
            dropout (float): Dropout概率
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 词嵌入层，将词汇索引转换为稠密向量
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=Config.PAD_TOKEN)
        
        # LSTM层，可选择双向
        self.lstm = nn.LSTM(
            embedding_size, 
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,  # 只有多层时才使用dropout
            bidirectional=Config.BIDIRECTIONAL,
            batch_first=True
        )
        
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 如果是双向LSTM，需要投影层将双向输出合并为单向
        if Config.BIDIRECTIONAL:
            self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.cell_projection = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, input_seq, input_lengths=None):
        """
        编码器前向传播
        
        Args:
            input_seq (Tensor): 输入序列，形状为(batch_size, seq_len)
            input_lengths (Tensor, optional): 每个序列的真实长度
            
        Returns:
            tuple: (LSTM输出, (最终隐藏状态, 最终细胞状态))
        """
        batch_size = input_seq.size(0)
        
        # 词嵌入：(batch_size, seq_len) -> (batch_size, seq_len, embedding_size)
        embedded = self.embedding(input_seq)
        embedded = self.dropout(embedded)
        
        # LSTM前向传播
        if input_lengths is not None:
            # 如果提供了序列长度，使用pack_padded_sequence提高效率
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths, batch_first=True, enforce_sorted=False
            )
            outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            # 直接处理填充后的序列
            outputs, (hidden, cell) = self.lstm(embedded)
        
        # 如果是双向LSTM，需要合并前向和后向的隐藏状态
        if Config.BIDIRECTIONAL:
            # hidden/cell形状: (num_layers * 2, batch_size, hidden_size)
            # 重新整形为: (num_layers, 2, batch_size, hidden_size)
            hidden = hidden.view(self.num_layers, 2, batch_size, -1)
            # 连接前向和后向: (num_layers, batch_size, hidden_size * 2)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            # 投影到原始隐藏维度: (num_layers, batch_size, hidden_size)
            hidden = self.hidden_projection(hidden)
            
            # 对细胞状态做相同处理
            cell = cell.view(self.num_layers, 2, batch_size, -1)
            cell = torch.cat([cell[:, 0, :, :], cell[:, 1, :, :]], dim=2)
            cell = self.cell_projection(cell)
        
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    """
    解码器类，使用LSTM基于编码器的输出逐步生成目标序列
    支持Teacher Forcing训练策略
    """
    
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
        """
        初始化解码器
        
        Args:
            output_size (int): 输出词汇表大小
            embedding_size (int): 词嵌入维度
            hidden_size (int): LSTM隐藏层大小
            num_layers (int): LSTM层数
            dropout (float): Dropout概率
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=Config.PAD_TOKEN)
        
        # LSTM层（单向）
        self.lstm = nn.LSTM(
            embedding_size, 
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 输出投影层，将隐藏状态映射到词汇表概率分布
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_token, hidden, cell):
        """
        解码器单步前向传播
        
        Args:
            input_token (Tensor): 当前输入token，形状为(batch_size, 1)
            hidden (Tensor): LSTM隐藏状态，形状为(num_layers, batch_size, hidden_size)
            cell (Tensor): LSTM细胞状态，形状为(num_layers, batch_size, hidden_size)
            
        Returns:
            tuple: (输出概率分布, 新隐藏状态, 新细胞状态)
        """
        # 词嵌入：(batch_size, 1) -> (batch_size, 1, embedding_size)
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        # LSTM前向传播：输入上一时刻的隐藏状态和细胞状态
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output形状: (batch_size, 1, hidden_size)
        
        # 投影到词汇表：(batch_size, hidden_size) -> (batch_size, output_size)
        prediction = self.out(output.squeeze(1))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """
    完整的Seq2Seq模型，结合编码器和解码器
    实现序列到序列的翻译功能
    """
    
    def __init__(self, encoder, decoder):
        """
        初始化Seq2Seq模型
        
        Args:
            encoder (Encoder): 编码器实例
            decoder (Decoder): 解码器实例
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        训练时的前向传播
        
        Args:
            src (Tensor): 源序列，形状为(batch_size, src_len)
            trg (Tensor): 目标序列，形状为(batch_size, trg_len)
            teacher_forcing_ratio (float): Teacher forcing使用概率
            
        Returns:
            Tensor: 解码器输出，形状为(batch_size, trg_len, trg_vocab_size)
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # 初始化输出张量，存储所有时刻的解码器输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(Config.DEVICE)
        
        # 编码器处理源序列
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # 解码器的第一个输入是SOS标记
        input_token = trg[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        # 逐步解码目标序列
        for t in range(1, trg_len):
            # 解码器单步前向传播
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            
            # 存储当前时刻的输出
            outputs[:, t, :] = output
            
            # Teacher Forcing策略：随机选择使用真实标签还是预测结果
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)  # 获取概率最高的词汇索引
            
            if teacher_force:
                # 使用真实标签作为下一个输入（Teacher Forcing）
                input_token = trg[:, t].unsqueeze(1)
            else:
                # 使用模型预测结果作为下一个输入
                input_token = top1.unsqueeze(1)
        
        return outputs
    
    def translate(self, src, max_length=50):
        """
        推理时的翻译函数，生成目标序列
        
        Args:
            src (Tensor): 源序列，形状为(batch_size, src_len)
            max_length (int): 生成序列的最大长度
            
        Returns:
            Tensor: 生成的目标序列，形状为(batch_size, seq_len)
        """
        self.eval()  # 切换到评估模式
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # 编码源序列
            encoder_outputs, (hidden, cell) = self.encoder(src)
            
            # 解码器初始输入为SOS标记
            input_token = torch.tensor([[Config.SOS_TOKEN]] * batch_size).to(Config.DEVICE)
            
            outputs = []
            # 逐步生成目标序列
            for _ in range(max_length):
                # 解码器单步前向传播
                output, hidden, cell = self.decoder(input_token, hidden, cell)
                
                # 选择概率最高的词汇
                top1 = output.argmax(1)
                outputs.append(top1.cpu().numpy())
                
                # 如果所有序列都生成了EOS标记，提前停止
                if all(token == Config.EOS_TOKEN for token in top1):
                    break
                
                # 使用预测结果作为下一个输入
                input_token = top1.unsqueeze(1)
            
            # 转置输出：(seq_len, batch_size) -> (batch_size, seq_len)
            return torch.tensor(outputs).transpose(0, 1)

def create_model(input_vocab_size, output_vocab_size):
    """
    创建完整的Seq2Seq模型
    
    Args:
        input_vocab_size (int): 输入语言词汇表大小
        output_vocab_size (int): 输出语言词汇表大小
        
    Returns:
        Seq2Seq: 创建好的模型实例
    """
    # 创建编码器
    encoder = Encoder(
        input_vocab_size,
        Config.EMBEDDING_SIZE,
        Config.HIDDEN_SIZE,
        Config.NUM_LAYERS,
        Config.DROPOUT
    )
    
    # 创建解码器
    decoder = Decoder(
        output_vocab_size,
        Config.EMBEDDING_SIZE,
        Config.HIDDEN_SIZE,
        Config.NUM_LAYERS,
        Config.DROPOUT
    )
    
    # 组合成完整模型并移动到指定设备
    model = Seq2Seq(encoder, decoder).to(Config.DEVICE)
    return model

def count_parameters(model):
    """
    计算模型的可训练参数数量
    
    Args:
        model (nn.Module): PyTorch模型
        
    Returns:
        int: 可训练参数的总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 