import json
import random
import argparse

def shuffle_jsonl(input_path, output_path):
    # 读取原始JSONL文件
    with open(input_path, 'r', encoding='utf-8') as f:
        # 逐行加载所有数据条目
        data = [json.loads(line) for line in f]
    
    # 使用Fisher-Yates算法原地打乱顺序
    random.shuffle(data)
    
    # 写入新的JSONL文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            # 确保每行独立序列化并添加换行符（符合JSONL规范）
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(args):
    # 输入文件名
    input_filename = args.input_filename
    # 输出文件名
    output_filename = args.output_filename
    # 随机洗牌
    shuffle_jsonl(input_filename, output_filename)
    print(f"数据集: {output_filename}")

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
        default="test_dataset_shuffle.jsonl"
    )
    main(parser.parse_args())