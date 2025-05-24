"""
词汇表管理模块
用于构建和管理英文和中文的词汇表，包括词汇到索引的映射、文本预处理等功能
"""

import re
import jieba        # 中文分词库
import pickle       # 用于序列化词汇表
from config import Config

class Vocabulary:
    """
    词汇表类，用于管理语言的词汇表
    支持英文和中文，提供词汇与索引的双向映射
    """
    
    def __init__(self, name):
        """
        初始化词汇表
        
        Args:
            name (str): 词汇表名称，'english'或'chinese'
        """
        self.name = name
        
        # 词汇到索引的映射字典，初始化包含特殊标记
        self.word2index = {
            '<SOS>': Config.SOS_TOKEN,    # 句子开始标记
            '<EOS>': Config.EOS_TOKEN,    # 句子结束标记
            '<PAD>': Config.PAD_TOKEN,    # 填充标记
            '<UNK>': Config.UNK_TOKEN     # 未知词标记
        }
        
        # 词汇出现次数统计
        self.word2count = {}
        
        # 索引到词汇的映射字典
        self.index2word = {
            Config.SOS_TOKEN: '<SOS>',
            Config.EOS_TOKEN: '<EOS>',
            Config.PAD_TOKEN: '<PAD>',
            Config.UNK_TOKEN: '<UNK>'
        }
        
        # 词汇表大小，初始为4（包含4个特殊标记）
        self.n_words = 4
        
    def add_sentence(self, sentence):
        """
        将句子中的词汇添加到词汇表
        
        Args:
            sentence (str): 输入句子
        """
        if self.name == 'chinese':
            # 中文使用jieba分词
            words = list(jieba.cut(sentence.strip()))
        else:  # English
            # 英文先标准化再按空格分词
            words = self.normalize_string(sentence).split(' ')
        
        # 将每个词添加到词汇表
        for word in words:
            self.add_word(word)
    
    def add_word(self, word):
        """
        向词汇表添加单个词汇
        
        Args:
            word (str): 要添加的词汇
        """
        if word not in self.word2index:
            # 新词汇：分配新索引
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            # 已存在词汇：增加计数
            self.word2count[word] += 1
    
    def normalize_string(self, s):
        """
        英文文本标准化处理
        
        Args:
            s (str): 原始英文文本
            
        Returns:
            str: 标准化后的文本
        """
        # 转换为小写并去除首尾空格
        s = s.lower().strip()
        
        # 在标点符号前添加空格，便于分词
        s = re.sub(r"([.!?])", r" \1", s)
        
        # 移除非字母和特殊标点的字符
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        
        return s
    
    def sentence_to_indexes(self, sentence):
        """
        将句子转换为索引序列
        
        Args:
            sentence (str): 输入句子
            
        Returns:
            list: 索引序列
        """
        if self.name == 'chinese':
            # 中文分词
            words = list(jieba.cut(sentence.strip()))
        else:
            # 英文标准化后分词
            words = self.normalize_string(sentence).split(' ')
        
        indexes = []
        for word in words:
            if word in self.word2index:
                # 词汇在词汇表中，使用对应索引
                indexes.append(self.word2index[word])
            else:
                # 词汇不在词汇表中，使用UNK标记
                indexes.append(self.word2index['<UNK>'])
        return indexes
    
    def indexes_to_sentence(self, indexes):
        """
        将索引序列转换回句子
        
        Args:
            indexes (list): 索引序列
            
        Returns:
            str: 转换后的句子
        """
        words = []
        for index in indexes:
            # 遇到EOS标记时停止
            if index == Config.EOS_TOKEN:
                break
            # 跳过SOS和PAD标记
            if index not in [Config.SOS_TOKEN, Config.PAD_TOKEN]:
                words.append(self.index2word[index])
        
        # 英文用空格连接，中文直接连接
        return ' '.join(words) if self.name == 'english' else ''.join(words)
    
    def trim_vocab(self, min_count):
        """
        修剪词汇表，移除低频词汇
        
        Args:
            min_count (int): 最小出现次数阈值
        """
        # 筛选出现次数大于等于阈值的词汇
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        
        print(f'保留词汇: {len(keep_words)} / {len(self.word2index)}')
        
        # 重新构建词汇表，保留特殊标记
        self.word2index = {
            '<SOS>': Config.SOS_TOKEN,
            '<EOS>': Config.EOS_TOKEN,
            '<PAD>': Config.PAD_TOKEN,
            '<UNK>': Config.UNK_TOKEN
        }
        self.index2word = {
            Config.SOS_TOKEN: '<SOS>',
            Config.EOS_TOKEN: '<EOS>',
            Config.PAD_TOKEN: '<PAD>',
            Config.UNK_TOKEN: '<UNK>'
        }
        self.n_words = 4
        
        # 重新添加保留的词汇
        for word in keep_words:
            self.add_word(word)
    
    def save(self, filepath):
        """
        保存词汇表到文件
        
        Args:
            filepath (str): 保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """
        从文件加载词汇表
        
        Args:
            filepath (str): 文件路径
            
        Returns:
            Vocabulary: 加载的词汇表对象
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f) 