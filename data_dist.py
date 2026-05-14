import json
import argparse
import statistics
import langid
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

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
    # 输入文件名
    input_filename = args.input_filename
    # 输出文件名
    output_filename = args.output_filename
    # 绘制分布图
    plot_distribution(input_filename, output_filename)
    print(f"分布图: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_filename",
        type=str,
        default="test_dataset.jsonl"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="test_dataset.png"
    )
    main(parser.parse_args())