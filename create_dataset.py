import os
import ast
import math
import json
import random
import bisect
import argparse
import statistics
import langid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from collections import Counter

def distr_type(value):
    """自定义类型校验函数，验证字典格式和数值规范"""
    try:
        # 先尝试JSON解析（处理双引号字符串键）
        data = json.loads(value.replace("'", '"'))  # 统一替换单引号为双引号
    except json.JSONDecodeError:
        try:
            # 再尝试Python字面量解析（处理无引号整数键）
            data = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            print(f"--distribution {value}: 格式应为'{{\"input长度1\": 比例1, \"input长度2\": 比例2, ...}}' (比例总和1.0)")
            raise ValueError()

    total = 0.0
    for k, v in data.items():
        # 校验键是否为整数
        if not str(k).isdigit():
            print(f"--distribution {value}: 长度 {k} 不是有效的整数")
            raise ValueError()
            
        # 校验值是否为小数
        if not isinstance(v, (int, float)):
            print(f"--distribution {value}: 比例 {v} 不是有效的数值")
            raise ValueError()
        total += float(v)
        
    # 校验比例总和为1（考虑浮点精度）
    if not math.isclose(total, 1.0, rel_tol=1e-3):
        print(f"--distribution {value}: 比例总和 {total:.3f} 不等于1")
        raise ValueError()
    
    return data

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

def check_normality(test_dataset, alpha=0.05):
    """
    综合判断数据是否符合正态分布
    参数：
        data : 待检验数据数组
        alpha : 显著性水平(默认0.05)
    返回：
        检验结果
    """
    # 执行统计检验
    data = [item["len"] for item in test_dataset]
    print("\n=== 统计检验结果 ===")
    # Shapiro-Wilk检验（适用于小样本）
    if len(data) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
        print(f"Shapiro-Wilk检验: p值={shapiro_p:.4f}", 
              "-> 符合正态分布" if shapiro_p > alpha else "-> 非正态分布")
    
    # D'Agostino's K-squared检验（适用于大样本）
    dagostino_stat, dagostino_p = stats.normaltest(data)
    print(f"D'Agostino检验: p值={dagostino_p:.4f}",
          "-> 符合正态分布" if dagostino_p > alpha else "-> 非正态分布")
    
    # 综合判断建议
    final_judge = all([p_val > alpha for p_val in [shapiro_p, dagostino_p]]) if len(data)<=5000 else (dagostino_p > alpha)
    print(f"\n综合结论: 数据{'符合' if final_judge else '不符合'}正态分布（α={alpha}）\n")

    return final_judge

def find_closest_records(input_file, target_sequence, min_length, max_length):
    # 预处理目标序列
    sorted_seq = sorted(target_sequence)
    test_dataset = [{"dist":float("inf"), "len":0, "line":""} for _ in range(len(target_sequence))]

    # 流式读取文件并提取长度及条目
    records = []
    total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
    with open(input_file, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, total=total_lines, desc="读取数据全集"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过无效的JSON行
            if record.get("answer") is None:
                record["answer"] = ""
            token_len = record.get("question_token_len")
            if token_len is None:
                continue  # 跳过缺失length字段的行
            if token_len < min_length or token_len > max_length:
                continue  # 最小最大长度过滤
            records.append((token_len, record))
    
    # 按长度排序条目
    records.sort(key=lambda x: x[0])
    lengths = [e[0] for e in records]

     # 初始化使用标记数组
    used = [False] * len(records)
    test_dataset = []

    # 为每个数值查找最近条目
    for number in tqdm(sorted_seq, desc="挑选合适数据"):
        pos = bisect.bisect_left(lengths, number)
        left, right = pos - 1, pos
        closest_idx = None
        min_diff = float("inf")
        
        # 双向扫描寻找最近可用条目
        while left >= 0 or right < len(records):
            candidates = []
            if left >= 0:
                diff_left = abs(records[left][0] - number)
                candidates.append((left, diff_left))
            if right < len(records):
                diff_right = abs(records[right][0] - number)
                candidates.append((right, diff_right))
            
            if not candidates:
                break  # 理论上不会触发
            
            # 优先选择差值更小的候选
            candidates.sort(key=lambda x: (x[1], x[0]))
            
            # 检查候选条目可用性
            for idx, diff in candidates:
                if not used[idx] and diff < min_diff:
                    closest_idx = idx
                    min_diff = diff
                    break
            if closest_idx is not None:
                break
            else:
                # 扩展搜索范围
                left -= 1
                right += 1
        
        # 记录找到的条目
        if closest_idx is not None:
            used[closest_idx] = True
            test_dataset.append({"len":records[closest_idx][0], "line":records[closest_idx][1]})

    return test_dataset

def save_output(test_dataset, output_file):
    # 写入结果文件
    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in test_dataset:
                outfile.write(json.dumps(item["line"], ensure_ascii=False) + "\n")

def main(args):
    # 输入长度
    input_len = args.input_len
    # 标准差缩放
    variance_scale = args.variance_scale
    # 输出样本数量
    max_lines = args.max_lines
    # 输出样本比例
    distr = args.distribution
    # 输入文件名
    input_filename = args.input_filename
    # 输出文件名
    output_filename = args.output_filename
    # 最小长度
    min_length = args.min_length
    # 最大长度
    max_length = args.max_length

    if distr:
        variance_scale = 0.0
        # 生成样本比例序列
        target_seq = [int(key) for key, ratio in distr.items() for _ in range(math.ceil(ratio * max_lines))]
        random.shuffle(target_seq)
    else:
        # 生成正态分布序列
        target_seq = np.random.normal(input_len, (input_len / 4) * variance_scale, max_lines)

    # 挑选符合分布的数据
    test_dataset = find_closest_records(input_filename, target_seq[:max_lines], min_length, max_length)
    if len(test_dataset) < max_lines:
        print(f"符合长度范围的实际数据条数({len(test_dataset)})小于期望条数({max_lines})!")
    # 随机洗牌
    random.shuffle(test_dataset)
    # 保存结果
    save_output(test_dataset, output_filename)
    print(f"数据集: {output_filename}")

    if variance_scale != 0.0:
        # 正态性检验
        check_normality(test_dataset)
    
    # 数据分布可视化
    distribution_filename = f"{os.path.splitext(output_filename)[0]}.png"
    plot_distribution(output_filename, distribution_filename)
    print(f"分布图: {distribution_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_len", type=int, default=1024)
    parser.add_argument("--variance_scale", type=float, default=1)
    parser.add_argument("--max_lines", type=int, default=500)
    parser.add_argument("--distribution", type=distr_type, default={})
    parser.add_argument("--min_length", type=float, default=0)
    parser.add_argument("--max_length", type=float, default=float('inf'))
    parser.add_argument(
        "--input_filename",
        type=str,
        default="merged_dataset.jsonl"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="test_dataset.jsonl"
    )
    main(parser.parse_args())