from math import remainder
import os
import json
import random
import langid
import argparse
import statistics
import multiprocessing
import numpy as np
import jionlp as jio
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from functools import partial
from collections import Counter
from transformers import AutoTokenizer

model_path = os.path.abspath("./DeepSeekR1/")

class TokenizerWrapper:
    """解决多进程tokenizer共享问题"""
    _instance = None
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        
    @classmethod
    def get_tokenizer(cls):
        if not cls._instance:
            cls._instance = TokenizerWrapper()
        return cls._instance.tokenizer

def count_token_len(text):
    """Token统计函数（兼容多进程）"""
    return len(TokenizerWrapper.get_tokenizer().encode(text))

def process_line(line, aug_scale, min_length, max_length):
    """处理单行数据（子进程安全版本）"""
    try:
        data = json.loads(line.strip())
        text = data.get("question", "")
        
        # 确保处理文本有效性
        if not isinstance(text, str) or len(text) == 0:
            return []
            
        records = []
        # 原始数据
        if data.get('question_token_len') is None:
            data['question_token_len'] = count_token_len(text)
        records.append(json.dumps(data, ensure_ascii=False))
        
        # 数据增强
        target_num = aug_scale - 1  # 需要生成的增强总数
        homophone_num  = max(int(target_num ** (1/3)), 1)
        homophone_texts = jio.homophone_substitution(text, augmentation_num=homophone_num) or [text]
        
        swap_num = max(int((target_num // len(homophone_texts)) ** (1/2)), 1)
        for homophone in homophone_texts:
            swapped_texts = jio.swap_char_position(homophone, augmentation_num=swap_num) or [homophone]
            
            random_num = max(int(target_num // (len(homophone_texts)*len(swapped_texts))), 1)
            for swapped in swapped_texts:
                random_texts = jio.random_add_delete(swapped, augmentation_num=random_num, add_ratio=0) or [swapped]
                
                for random in random_texts:
                    token_len = count_token_len(random)
                    if token_len < min_length or token_len > max_length: 
                        continue  # 最小最大长度过滤
                    new_data = data.copy()
                    new_data["question"] = random
                    new_data['question_token_len'] = token_len
                    records.append(json.dumps(new_data, ensure_ascii=False))

        remainder = aug_scale - len(records)
        for _ in range(remainder):
            current_text = text
            # 各步增强增加有效性检查
            for aug_func in [
                jio.homophone_substitution,
                jio.swap_char_position,
                jio.random_add_delete,
            ]:
                try:
                    # 根据不同的增强函数构造参数字典
                    kwargs = {"augmentation_num": 1}
                    if aug_func == jio.random_add_delete:
                        kwargs["add_ratio"] = 0  # 添加额外参数

                    aug_result = aug_func(current_text, **kwargs)

                    if aug_result and isinstance(aug_result, list):
                        current_text = aug_result[0]
                except:
                    pass  # 增强失败时保持原文本
            
            token_len = count_token_len(current_text)
            if token_len < min_length or token_len > max_length: 
                continue  # 最小最大长度过滤
            new_data = data.copy()
            new_data["question"] = current_text
            new_data['question_token_len'] = token_len
            records.append(json.dumps(new_data, ensure_ascii=False))

        return records
    except Exception as e:
        print(f"Error processing line: {e}")
        return []

def data_augmentation(lines, aug_scale, min_length, max_length):
    """并行数据增强"""
    process_func = partial(process_line, aug_scale=aug_scale, min_length=min_length, max_length=max_length)
    with multiprocessing.Pool(os.cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_func, lines),
            total=len(lines),
            desc="处理进度",
        ))
    return [line for sublist in results for line in sublist]

def plot_distribution(input_file, output_file):
    records = []
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            token_len = record.get("question_token_len")
            if token_len is None:
                continue
            lang = record.get("question_lang")
            if lang is None:
                question = record.get("question")
                if question is None:
                    lang = ''
                else:
                    lang = langid.classify(question)[0]
            records.append((token_len, lang))

    # 统计token长度
    lengths = [x[0] for x in records]
    # 统计各语言出现频次
    lang_counts = Counter([x[1] for x in records])
    labels, values = zip(*lang_counts.items())  # 解压为标签和频次列表

    plt.switch_backend("agg")
    # 创建画布
    plt.figure(figsize=(12, 5))
    
    # 子图1：直方图+密度曲线
    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=30, alpha=0.6, color="g")
    
    # 绘制正态分布曲线
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(lengths), np.std(lengths))  # 正态分布概率密度
    bin_width = (xmax - xmin) / 30  # 计算每个bin的宽度
    plt.plot(x, p * len(lengths) * bin_width, "k", linewidth=2)  # 调整曲线高度匹配样本数量
    plt.title(f"Histogram with Normal Curve\nMin: {min(lengths)}, Max: {max(lengths)}, Mean: {statistics.mean(lengths)}")
    print(f"最小值: {min(lengths)}, 最大值: {max(lengths)}, 均值: {statistics.mean(lengths)}")
    
    # 绘制条形图（适用于分类数据）
    plt.subplot(1, 2, 2)
    plt.bar(range(len(labels)), values, tick_label=labels, edgecolor='black')
    
    # 添加图表标注
    plt.title('Language Distribution in JSONL File')
    plt.xlabel('Language Code')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # 旋转标签避免重叠
    plt.grid(axis='y', alpha=0.5)
    
    plt.tight_layout()  # 自动调整布局
    plt.savefig(output_file)  # 保存图片
    plt.close()  # 关闭图形释放内存

def main(args):
    # 数据扩充倍数
    aug_scale = args.aug_scale
    # 最小长度
    min_length = args.min_length
    # 最大长度
    max_length = args.max_length
    # 输入文件名
    input_filename = args.input_filename
    # 输出文件名
    output_filename = args.output_filename
    # 数据扩充
    lines = open(input_filename, 'r', encoding='utf-8').readlines()
    data_aug = data_augmentation(lines, aug_scale, min_length, max_length)
    # 扩充条数不足提醒
    if len(data_aug) < len(lines) * aug_scale:
        print(f"实际扩充条数{len(data_aug)}小于期望条数{len(lines) * aug_scale}")
    # 随机洗牌
    random.shuffle(data_aug)
    # 保存结果
    open(output_filename, 'w', encoding='utf-8').write('\n'.join(data_aug))
    print(f"数据集: {output_filename}")
    # 数据分布可视化
    distribution_filename = f"{os.path.splitext(output_filename)[0]}.png"
    plot_distribution(output_filename, distribution_filename)
    print(f"分布图: {distribution_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_scale", type=int, default=10)
    parser.add_argument("--min_length", type=float, default=0)
    parser.add_argument("--max_length", type=float, default=float('inf'))
    parser.add_argument(
        "--input_filename",
        type=str,
        default="test_dataset.jsonl"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="test_dataset_aug.jsonl"
    )
    # Windows多进程保护
    multiprocessing.freeze_support()
    main(parser.parse_args())