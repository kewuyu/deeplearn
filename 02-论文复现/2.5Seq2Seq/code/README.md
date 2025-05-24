# Seq2Seq 机器翻译系统

这是一个基于PyTorch实现的Seq2Seq机器翻译系统，使用CMN（中英文）数据集训练英文到中文的翻译模型。

## 项目特点

- **原始Seq2Seq架构**: 使用编码器-解码器架构，包含LSTM层
- **双向编码器**: 支持双向LSTM编码器以更好地理解输入序列
- **Teacher Forcing**: 训练时使用teacher forcing技术加速收敛
- **词汇表管理**: 自动处理英文和中文词汇表构建
- **数据预处理**: 完整的数据清洗和预处理流程
- **模型保存与加载**: 支持训练过程中的模型保存和后续加载

## 目录结构

```
.
├── cmn.txt              # CMN数据集文件
├── config.py            # 配置文件
├── vocabulary.py        # 词汇表管理
├── data_loader.py       # 数据加载和预处理
├── models.py            # Seq2Seq模型定义
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型架构

### 编码器 (Encoder)
- 嵌入层：将输入词汇转换为向量表示
- 双向LSTM：处理输入序列，获得上下文表示
- 隐藏状态投影：将双向LSTM的隐藏状态合并

### 解码器 (Decoder)
- 嵌入层：处理目标语言词汇
- 单向LSTM：基于编码器状态生成目标序列
- 输出层：将隐藏状态映射到词汇表概率分布

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程中会：
- 自动预处理CMN数据集
- 构建英文和中文词汇表
- 训练Seq2Seq模型
- 定期保存模型检查点
- 显示训练进度和损失曲线
- 在验证集上评估模型性能

### 2. 模型推理

#### 交互式翻译
```bash
python inference.py --mode interactive
```

#### 批量翻译
```bash
python inference.py --mode batch --input_file input.txt --output_file output.txt
```

#### 演示翻译
```bash
python inference.py --mode demo
```

## 配置参数

在 `config.py` 中可以调整以下参数：

### 数据参数
- `MAX_LENGTH`: 最大句子长度 (默认: 30)
- `MIN_COUNT`: 词汇最小出现次数 (默认: 2)

### 模型参数
- `HIDDEN_SIZE`: LSTM隐藏层大小 (默认: 256)
- `EMBEDDING_SIZE`: 嵌入层维度 (默认: 256)
- `NUM_LAYERS`: LSTM层数 (默认: 2)
- `DROPOUT`: Dropout比率 (默认: 0.1)
- `BIDIRECTIONAL`: 是否使用双向编码器 (默认: True)

### 训练参数
- `BATCH_SIZE`: 批次大小 (默认: 64)
- `LEARNING_RATE`: 学习率 (默认: 0.001)
- `NUM_EPOCHS`: 训练轮数 (默认: 50)

## 数据集

使用CMN数据集，包含约30,000个英中句子对。数据格式为：
```
English sentence    Chinese sentence
```

## 训练输出

训练过程中会生成以下文件：
- `best_seq2seq_model.pth`: 最佳模型权重
- `input_vocab.pkl`: 英文词汇表
- `output_vocab.pkl`: 中文词汇表
- `loss_curve.png`: 训练损失曲线图

