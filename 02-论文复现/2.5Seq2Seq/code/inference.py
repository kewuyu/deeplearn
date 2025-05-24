"""
Seq2Seq机器翻译推理脚本
支持交互式翻译、批量翻译和演示翻译三种模式
加载训练好的模型进行英文到中文的翻译
"""

import torch
import argparse
from models import create_model
from vocabulary import Vocabulary
from config import Config

def load_model_and_vocabs(model_path, input_vocab_path, output_vocab_path):
    """
    加载训练好的模型和词汇表
    
    Args:
        model_path (str): 模型权重文件路径
        input_vocab_path (str): 输入语言词汇表路径
        output_vocab_path (str): 输出语言词汇表路径
        
    Returns:
        tuple: (模型实例, 输入词汇表, 输出词汇表)
    """
    # 加载词汇表
    input_vocab = Vocabulary.load(input_vocab_path)
    output_vocab = Vocabulary.load(output_vocab_path)
    
    # 根据词汇表大小创建模型
    model = create_model(input_vocab.n_words, output_vocab.n_words)
    
    # 加载训练好的模型权重
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    
    return model, input_vocab, output_vocab

def translate_sentence(model, sentence, input_vocab, output_vocab, max_length=50):
    """
    翻译单个英文句子
    
    Args:
        model (Seq2Seq): 训练好的翻译模型
        sentence (str): 待翻译的英文句子
        input_vocab (Vocabulary): 输入语言词汇表
        output_vocab (Vocabulary): 输出语言词汇表
        max_length (int): 生成序列的最大长度
        
    Returns:
        str: 翻译后的中文句子
    """
    model.eval()
    
    with torch.no_grad():
        # 将句子转换为索引序列
        tokens = input_vocab.sentence_to_indexes(sentence)
        # 添加句子开始和结束标记
        tokens = [Config.SOS_TOKEN] + tokens + [Config.EOS_TOKEN]
        
        # 转换为张量并添加批次维度
        src_tensor = torch.tensor(tokens).unsqueeze(0).to(Config.DEVICE)
        
        # 使用模型进行翻译
        trg_indexes = model.translate(src_tensor, max_length)
        
        # 将索引序列转换回中文文本
        trg_tokens = trg_indexes[0].tolist()
        return output_vocab.indexes_to_sentence(trg_tokens)

def interactive_translation():
    """
    交互式翻译模式
    用户可以持续输入英文句子进行翻译，直到输入quit退出
    """
    # 定义模型文件路径
    model_path = f'best_{Config.MODEL_SAVE_PATH}'
    input_vocab_path = f'input_{Config.VOCAB_SAVE_PATH}'
    output_vocab_path = f'output_{Config.VOCAB_SAVE_PATH}'
    
    try:
        print("正在加载模型...")
        model, input_vocab, output_vocab = load_model_and_vocabs(
            model_path, input_vocab_path, output_vocab_path
        )
        print("模型加载成功!")
        print("=" * 50)
        print("英中翻译系统")
        print("输入 'quit' 或 'exit' 退出程序")
        print("=" * 50)
        
        # 交互循环
        while True:
            # 获取用户输入
            english_text = input("\n请输入英文句子: ").strip()
            
            # 检查退出条件
            if english_text.lower() in ['quit', 'exit', '退出']:
                print("再见!")
                break
            
            # 检查输入有效性
            if not english_text:
                print("请输入有效的英文句子!")
                continue
            
            # 执行翻译
            try:
                chinese_translation = translate_sentence(
                    model, english_text, input_vocab, output_vocab
                )
                print(f"中文翻译: {chinese_translation}")
            except Exception as e:
                print(f"翻译出错: {e}")
                
    except FileNotFoundError as e:
        print(f"找不到模型文件: {e}")
        print("请先运行训练脚本生成模型文件")
    except Exception as e:
        print(f"加载模型时出错: {e}")

def batch_translation(input_file, output_file):
    """
    批量翻译模式
    读取文件中的英文句子，逐行翻译并保存到输出文件
    
    Args:
        input_file (str): 输入文件路径，每行一个英文句子
        output_file (str): 输出文件路径，保存翻译结果
    """
    model_path = f'best_{Config.MODEL_SAVE_PATH}'
    input_vocab_path = f'input_{Config.VOCAB_SAVE_PATH}'
    output_vocab_path = f'output_{Config.VOCAB_SAVE_PATH}'
    
    try:
        print("正在加载模型...")
        model, input_vocab, output_vocab = load_model_and_vocabs(
            model_path, input_vocab_path, output_vocab_path
        )
        print("模型加载成功!")
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        
        print(f"开始翻译 {len(sentences)} 个句子...")
        
        # 翻译并写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if sentence:  # 跳过空行
                    translation = translate_sentence(
                        model, sentence, input_vocab, output_vocab
                    )
                    # 输出格式：原文\t译文
                    f.write(f"{sentence}\t{translation}\n")
                    
                    # 显示进度
                    if (i + 1) % 100 == 0:
                        print(f"已翻译 {i + 1}/{len(sentences)} 个句子")
        
        print(f"翻译完成! 结果保存到: {output_file}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"批量翻译出错: {e}")

def demo_translation():
    """
    演示翻译模式
    使用预定义的测试句子展示翻译效果
    """
    model_path = f'best_{Config.MODEL_SAVE_PATH}'
    input_vocab_path = f'input_{Config.VOCAB_SAVE_PATH}'
    output_vocab_path = f'output_{Config.VOCAB_SAVE_PATH}'
    
    # 预定义的测试句子
    test_sentences = [
        "Hello!",
        "Good morning.",
        "How are you?",
        "I love you.",
        "Thank you very much.",
        "What's your name?",
        "Nice to meet you.",
        "Have a good day.",
        "See you later.",
        "Good luck!"
    ]
    
    try:
        print("正在加载模型...")
        model, input_vocab, output_vocab = load_model_and_vocabs(
            model_path, input_vocab_path, output_vocab_path
        )
        print("模型加载成功!")
        
        print("\n=" * 60)
        print("翻译演示")
        print("=" * 60)
        
        # 逐个翻译测试句子
        for sentence in test_sentences:
            translation = translate_sentence(
                model, sentence, input_vocab, output_vocab
            )
            print(f"EN: {sentence}")
            print(f"ZH: {translation}")
            print("-" * 40)
            
    except FileNotFoundError as e:
        print(f"找不到模型文件: {e}")
        print("请先运行训练脚本生成模型文件")
    except Exception as e:
        print(f"演示翻译出错: {e}")

def main():
    """
    主函数，解析命令行参数并调用相应的翻译模式
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Seq2Seq机器翻译推理')
    parser.add_argument('--mode', choices=['interactive', 'batch', 'demo'], 
                       default='interactive', help='运行模式')
    parser.add_argument('--input_file', type=str, help='批量翻译输入文件')
    parser.add_argument('--output_file', type=str, help='批量翻译输出文件')
    
    # 解析参数
    args = parser.parse_args()
    
    # 根据模式调用相应函数
    if args.mode == 'interactive':
        # 交互式翻译模式
        interactive_translation()
    elif args.mode == 'batch':
        # 批量翻译模式
        if not args.input_file or not args.output_file:
            print("批量翻译模式需要指定输入和输出文件")
            print("使用方法: python inference.py --mode batch --input_file input.txt --output_file output.txt")
        else:
            batch_translation(args.input_file, args.output_file)
    elif args.mode == 'demo':
        # 演示翻译模式
        demo_translation()

if __name__ == '__main__':
    main() 