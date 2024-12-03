import os
from pathlib import Path
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

def categorize_steps(step):
    """将步数归类到指定的桶"""
    if step <= 7:
        return step
    elif 8 <= step <= 10:
        return 8
    else:
        return 9

def calculate_steps_and_accuracy(file_path):
    """从 JSON 文件中提取步数和正确率数据，同时统计步数分布"""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    steps_correct = defaultdict(list)
    steps_distribution = []  # 用于统计步数分布

    for entry in data:
        question = entry["question"]
        answer = entry["answer"]
        pred = entry["pred"]
        model_outputs = entry["model_outputs"]
        
        # 使用正则表达式计算最小推理步数
        sentences = re.split(r'\.\s+', model_outputs)
        answer_pattern = r"[aA]nswer is \(?([A-J])\)?"
        for i, sentence in enumerate(sentences):
            if re.search(answer_pattern, sentence):
                categorized_step = categorize_steps(i + 1)
                steps_correct[categorized_step].append(pred == answer)
                steps_distribution.append(categorized_step)
                break
    
    steps_accuracy = {}
    for step, correctness in steps_correct.items():
        steps_accuracy[step] = sum(correctness) / len(correctness)
    
    return steps_accuracy, steps_distribution

def plot_category_graph(category, steps_accuracy, save_path):
    """绘制单个 category 的图并保存"""
    steps = sorted(steps_accuracy.keys())
    accuracies = [steps_accuracy[step] for step in steps]
    
    plt.figure(figsize=(8, 6))
    plt.plot(steps, accuracies, marker='o', label=category)
    plt.title(f"Inference Steps vs Accuracy for {category}")
    plt.xlabel("Minimum Inference Steps (Buckets)")
    plt.ylabel("Accuracy")
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ["1", "2", "3", "4", "5", "6", "7", "8-10", ">10"])
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_steps_vs_accuracy(directory):
    """绘制推理步数与正确率的关系图"""
    all_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    category_data = {}
    steps_distributions = {}
    
    for file in all_files:
        file_path = os.path.join(directory, file)
        category = os.path.splitext(file)[0]
        steps_accuracy, steps_distribution = calculate_steps_and_accuracy(file_path)
        category_data[category] = steps_accuracy
        steps_distributions[category] = Counter(steps_distribution)
    
    # 打印步数分布
    for category, distribution in steps_distributions.items():
        print(f"Category: {category}")
        for step in sorted(distribution.keys()):
            bucket_label = (
                str(step) if step <= 7 else "8-10" if step == 8 else ">10"
            )
            print(f"  Steps {bucket_label}: {distribution[step]} questions")
        print()
    
    # 绘制每个 category 的图并保存
    save_path = Path(directory) / "step_accu_curve"
    os.makedirs(save_path, exist_ok=True)
    for category, steps_accuracy in category_data.items():
        plot_category_graph(category, steps_accuracy, save_path / f"{category}_steps_vs_accuracy.png")
    
    # 绘制综合图
    plt.figure(figsize=(12, 8))
    for category, steps_accuracy in category_data.items():
        steps = sorted(steps_accuracy.keys())
        accuracies = [steps_accuracy[step] for step in steps]
        plt.plot(steps, accuracies, marker='o', label=category)
    
    plt.title("Inference Steps vs Accuracy by Category")
    plt.xlabel("Minimum Inference Steps (Buckets)")
    plt.ylabel("Accuracy")
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ["1", "2", "3", "4", "5", "6", "7", "8-10", ">10"])
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path / 'all_categories.png')  # 保存综合图
    plt.show()

# 使用示例
json_directory = "/home/shaohanh/qilongma/MMLU-Pro/results/QwQ-32B-Preview/CoT/all"  # 替换为存放 JSON 文件的目录路径
plot_steps_vs_accuracy(json_directory)
