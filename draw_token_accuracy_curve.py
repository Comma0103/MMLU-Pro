import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re


MAX_NEW_TOKENS = 2048


def calculate_bins(values, num_bins):
    """根据输出 token 数生成均匀分布的桶边界"""
    return np.percentile(values, np.linspace(0, 100, num_bins + 1))

def categorize_into_bins(value, bins):
    """根据桶边界将值归入对应的桶"""
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i + 1
    return len(bins)  # 超过最后一个桶


def extract_answer(text):
    """
    提取答案字母和到该字母位置的 token 数量。
    """
    pattern = r"[aA]nswer is \(?([A-J])\)?"  # 匹配 Answer is (X) 格式
    match = re.search(pattern, text)
    if match:
        answer = match.group(1)
        tokens_count = len(re.split(r"\s+", text[:match.start()]))
        return answer, tokens_count
    else:
        # 如果第一个提取失败，进入第二种模式
        return extract_again(text)

def extract_again(text):
    """
    第二种模式提取答案，格式为 Answer: X。
    """
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        answer = match.group(1)
        tokens_count = len(re.split(r"\s+", text[:match.start()]))
        return answer, tokens_count
    else:
        # 如果第二种提取失败，进入第三种模式
        return extract_final(text)

def extract_final(text):
    """
    第三种模式提取答案，查找最后一个独立的大写字母 A-J。
    """
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"  # 匹配最后一个大写字母 A-J
    match = re.search(pattern, text, re.DOTALL)
    if match:
        answer = match.group(0)
        tokens_count = len(re.split(r"\s+", text[:match.start()]))
        return answer, tokens_count
    else:
        # 如果所有提取失败，返回 None
        print(text)
        return None, None

def calculate_tokens_and_accuracy(file_path, bins):
    """从 JSON 文件中提取 token 数和正确率数据，同时统计 token 分布"""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    tokens_correct = defaultdict(list)
    tokens_distribution = []  # 用于统计 token 分布

    for entry in data:
        question = entry["question"]
        answer = entry["answer"]
        pred = entry["pred"]
        model_outputs = entry["model_outputs"]

        # 使用正则表达式查找到答案为止
        # sentences = re.split(r'\.\s+', model_outputs)
        # answer_pattern = r"[aA]nswer is \(?([A-J])\)?"
        # token_count = 0
        # for sentence in sentences:
        #     token_count += len(sentence.split())  # 累加 token 数
        #     if re.search(answer_pattern, sentence):
        #         break
        _, token_count = extract_answer(model_outputs)
        if token_count is None:
            token_count = MAX_NEW_TOKENS

        bin_category = categorize_into_bins(token_count, bins)
        tokens_correct[bin_category].append(pred == answer)
        tokens_distribution.append(bin_category)
    
    tokens_accuracy = {}
    for token_bin, correctness in tokens_correct.items():
        tokens_accuracy[token_bin] = sum(correctness) / len(correctness)
    
    return tokens_accuracy, tokens_distribution

def plot_category_graph(category, tokens_accuracy, bins, save_path):
    """绘制单个 category 的图并保存"""
    token_bins = sorted(tokens_accuracy.keys())
    accuracies = [tokens_accuracy[token_bin] for token_bin in token_bins]
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
    ] + [f">{int(bins[-1])}"]

    plt.figure(figsize=(8, 6))
    plt.plot(token_bins, accuracies, marker='o', label=category)
    plt.title(f"Output Tokens vs Accuracy for {category}")
    plt.xlabel("Token Range (Buckets)")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_tokens_vs_accuracy(directory):
    """绘制输出 token 数与正确率的关系图，同时打印 token 分布"""
    all_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    all_token_counts = []  # 用于计算分桶策略
    category_data = {}
    tokens_distributions = {}

    # 统计所有问题的 token 数
    for file in all_files:
        file_path = os.path.join(directory, file)
        with open(file_path, "r") as f:
            data = json.load(f)
        for entry in data:
            model_outputs = entry["model_outputs"]

            # 计算到出现答案为止的 token 数
            # sentences = re.split(r'\.\s+', model_outputs)
            # answer_pattern = r"[aA]nswer is \(?([A-J])\)?"
            # token_count = 0
            # for sentence in sentences:
            #     token_count += len(sentence.split())
            #     if re.search(answer_pattern, sentence):
            #         break
            _, token_count = extract_answer(model_outputs)
            if token_count is None:
                token_count = MAX_NEW_TOKENS
            all_token_counts.append(token_count)

    # 计算均匀分布的桶边界
    bins = calculate_bins(all_token_counts, 10)

    # 计算每个分类的 token 与正确率
    for file in all_files:
        file_path = os.path.join(directory, file)
        category = os.path.splitext(file)[0]
        tokens_accuracy, tokens_distribution = calculate_tokens_and_accuracy(file_path, bins)
        category_data[category] = tokens_accuracy
        tokens_distributions[category] = Counter(tokens_distribution)
    
    # 打印 token 分布
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
    ] + [f">{int(bins[-1])}"]
    for category, distribution in tokens_distributions.items():
        print(f"Category: {category}")
        for token_bin in sorted(distribution.keys()):
            print(f"  Tokens {bin_labels[token_bin - 1]}: {distribution[token_bin]} questions")
        print()
    
    # 绘制每个 category 的图并保存
    save_path = Path(directory) / "token_accu_curve"
    os.makedirs(save_path, exist_ok=True)
    for category, tokens_accuracy in category_data.items():
        plot_category_graph(category, tokens_accuracy, bins, save_path / f"{category}_tokens_vs_accuracy.png")
    
    # 绘制综合图
    plt.figure(figsize=(12, 8))
    for category, tokens_accuracy in category_data.items():
        token_bins = sorted(tokens_accuracy.keys())
        accuracies = [tokens_accuracy[token_bin] for token_bin in token_bins]
        plt.plot(token_bins, accuracies, marker='o', label=category)
    
    plt.title("Output Tokens vs Accuracy by Category")
    plt.xlabel("Token Range (Buckets)")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path / 'all_categories_tokens_vs_accuracy.png')
    plt.show()

# 使用示例
json_directory = "/home/shaohanh/qilongma/MMLU-Pro/results/QwQ-32B-Preview/CoT/all/0-shot"  # 替换为存放 JSON 文件的目录路径
plot_tokens_vs_accuracy(json_directory)
