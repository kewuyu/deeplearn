"""
Seq2Seq机器翻译系统配置文件
包含模型架构、训练参数、数据处理等所有配置项
"""

import torch

class Config:
    """
    配置类，包含所有训练和模型的超参数
    """
    
    # ==================== 数据相关配置 ====================
    DATA_PATH = "cmn.txt"          # CMN数据集文件路径
    MAX_LENGTH = 30                # 句子最大长度，超过此长度的句子将被过滤
    MIN_COUNT = 2                  # 词汇最小出现次数，低于此次数的词汇将被替换为UNK
    
    # ==================== 模型架构配置 ====================
    HIDDEN_SIZE = 256              # LSTM隐藏层大小
    EMBEDDING_SIZE = 256           # 词嵌入维度
    NUM_LAYERS = 2                 # LSTM层数
    DROPOUT = 0.1                  # Dropout概率，防止过拟合
    BIDIRECTIONAL = True           # 编码器是否使用双向LSTM
    
    # ==================== 训练相关配置 ====================
    BATCH_SIZE = 64                # 批次大小
    LEARNING_RATE = 0.001          # 学习率
    NUM_EPOCHS = 50                # 训练轮数
    PRINT_EVERY = 100              # 每多少个batch打印一次训练信息
    SAVE_EVERY = 5                 # 每多少个epoch保存一次模型
    
    # ==================== 设备配置 ====================
    # 自动选择GPU或CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== 特殊标记定义 ====================
    SOS_TOKEN = 0                  # 句子开始标记 (Start of Sentence)
    EOS_TOKEN = 1                  # 句子结束标记 (End of Sentence)
    PAD_TOKEN = 2                  # 填充标记 (Padding)
    UNK_TOKEN = 3                  # 未知词标记 (Unknown)
    
    # ==================== 文件保存路径 ====================
    MODEL_SAVE_PATH = "seq2seq_model.pth"    # 模型权重保存路径
    VOCAB_SAVE_PATH = "vocab.pkl"             # 词汇表保存路径
    
    # ==================== 数据集分割比例 ====================
    TRAIN_RATIO = 0.8              # 训练集比例
    VAL_RATIO = 0.1                # 验证集比例
    TEST_RATIO = 0.1               # 测试集比例 