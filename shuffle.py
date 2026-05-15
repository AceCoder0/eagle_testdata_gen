import json
import random
import argparse

def shuffle_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main(args):
    input_filename = args.input_filename
    output_filename = args.output_filename
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
