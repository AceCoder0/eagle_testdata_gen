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

    lengths = [x[0] for x in records]
    lang_counts = Counter([x[1] for x in records])
    labels, values = zip(*lang_counts.items())

    plt.switch_backend("agg")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=30, alpha=0.6, color="g")

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(lengths), np.std(lengths))
    bin_width = (xmax - xmin) / 30
    plt.plot(x, p * len(lengths) * bin_width, "k", linewidth=2)
    plt.title(f"Histogram with Normal Curve\nMin: {min(lengths)}, Max: {max(lengths)}, Mean: {statistics.mean(lengths)}")
    print(f"最小值: {min(lengths)}, 最大值: {max(lengths)}, 均值: {statistics.mean(lengths)}")

    plt.subplot(1, 2, 2)
    plt.bar(range(len(labels)), values, tick_label=labels, edgecolor='black')

    plt.title('Language Distribution in JSONL File')
    plt.xlabel('Language Code')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(args):
    input_filename = args.input_filename
    output_filename = args.output_filename
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
