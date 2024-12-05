import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def calculate_token_accuracy_and_distribution(file_path, bins, answer_pattern=r"[aA]nswer is \(?([A-J])\)?"):
    """
    计算每个桶内的准确率，以及所有问题的 token 数分布。
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    tokens_correct = defaultdict(list)
    token_distribution = defaultdict(int)

    for entry in data:
        question = entry["question"]
        answer = entry["answer"]
        pred = entry["pred"]
        model_outputs = entry["model_outputs"]

        # 计算输出 token 数直到匹配答案
        sentences = re.split(r'\.\s+', model_outputs)
        token_count = 0
        for sentence in sentences:
            token_count += len(sentence.split())
            if re.search(answer_pattern, sentence):
                break

        # 确定该问题的 token 数属于哪个桶
        bin_category = np.digitize([token_count], bins)[0]
        tokens_correct[bin_category].append(pred == answer)
        token_distribution[bin_category] += 1

    # 计算每个桶的准确率
    token_accuracy = {
        bin_idx: sum(correctness) / len(correctness)
        for bin_idx, correctness in tokens_correct.items()
    }
    return token_accuracy, token_distribution


def determine_bins_by_quantile(directories, num_bins=10):
    """
    根据所有模型的所有数据按分位数计算分桶边界。
    """
    all_token_counts = []
    for directory in directories:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
        for file_path in files:
            with open(file_path, "r") as f:
                data = json.load(f)
                for entry in data:
                    sentences = re.split(r'\.\s+', entry["model_outputs"])
                    token_count = 0
                    for sentence in sentences:
                        token_count += len(sentence.split())
                        if re.search(r"[aA]nswer is \(?([A-J])\)?", sentence):
                            break
                    all_token_counts.append(token_count)

    # 按分位数计算桶边界
    bins = np.quantile(all_token_counts, q=np.linspace(0, 1, num_bins + 1))
    return bins


def plot_category_graph(category, models_accuracy, bins, save_path):
    """
    绘制单个学科的所有模型 token-accuracy 曲线，并保存。
    """
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
    ] + [f">{int(bins[-1])}"]

    plt.figure(figsize=(10, 6))
    for model_name, tokens_accuracy in models_accuracy.items():
        token_bins = sorted(tokens_accuracy.keys())
        accuracies = [tokens_accuracy[token_bin] for token_bin in token_bins]
        plt.plot(token_bins, accuracies, marker='o', label=model_name)

    plt.title(f"Output Tokens vs Accuracy for {category}")
    plt.xlabel("Token Range (Buckets)")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(bin_labels) + 1), bin_labels, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_token_accuracy_by_category(directories, model_names, bins, output_dir="plots"):
    """
    按学科绘制所有模型的 token-accuracy 曲线，并打印 token 分布。
    """
    os.makedirs(output_dir, exist_ok=True)
    category_data = defaultdict(lambda: defaultdict(list))  # {category: {model: token_accuracy}}
    tokens_distributions = defaultdict(lambda: defaultdict(list))  # {category: {model: token_distribution}}

    # 计算每个模型的 token-accuracy 和分布
    for directory, model_name in zip(directories, model_names):
        files = [f for f in os.listdir(directory) if f.endswith(".json")]
        for file in files:
            file_path = os.path.join(directory, file)
            category = os.path.splitext(file)[0]
            token_accuracy, token_distribution = calculate_token_accuracy_and_distribution(file_path, bins)
            category_data[category][model_name] = token_accuracy
            tokens_distributions[category][model_name] = token_distribution

    # 打印 token 分布
    bin_labels = [
        f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)
    ] + [f">{int(bins[-1])}"]

    for category, models_distribution in tokens_distributions.items():
        print(f"Category: {category}")
        for model_name, distribution in models_distribution.items():
            print(f"  Model: {model_name}")
            for token_bin in sorted(distribution.keys()):
                print(f"    Tokens {bin_labels[token_bin - 1]}: {distribution[token_bin]} questions")
        print()

    # 绘制每个学科的图
    for category, models in category_data.items():
        save_path = os.path.join(output_dir, f"{category}_token_accuracy.png")
        plot_category_graph(category, models, bins, save_path)


# 使用示例
json_directories = [
    "/home/shaohanh/qilongma/MMLU-Pro/results/QwQ-32B-Preview/CoT/all",  # 替换为模型 1 的 JSON 文件目录路径
    "/home/shaohanh/qilongma/MMLU-Pro/results/Qwen2.5-32B-Instruct/CoT/all",  # 替换为模型 2 的 JSON 文件目录路径
    # 添加更多模型路径...
]
model_names = ["QwQ-32B-Preview", "Qwen2.5-32B-Instruct"]  # 替换为对应模型的名称

bins = determine_bins_by_quantile(json_directories, num_bins=10)
plot_token_accuracy_by_category(json_directories, model_names, bins, output_dir="results/multimodel_token_accu_curve")
