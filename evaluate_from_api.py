import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset

from call_gpt import Openai, API_INFOS
from crop import crop_prompt


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048
temperature = 0.0


def load_mmlu_pro():
    if args.data_path == "hf_hub":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    else:
        dataset = load_dataset("parquet", data_dir=args.data_path)
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    model_name = input_args.model.split("/")[-1]
    scoring_method = "CoT"
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    k = input_args.ntrain
    return [model_name, scoring_method, subjects, f"{k}-shot"]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?" # 在文本中找到以 answer is 开头，后面紧跟一个范围为 A-J 的字母（可选地被括号 () 包裹）的匹配项，并捕获字母作为结果。
    try:
        match = re.search(pattern, str(text))
    except Exception as e:
        logging.info(f"Error {str(e)} in extracting answer in response: " + text)
    if match:
        return match.group(1)
    else:
        logging.info("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text) # 最后一个匹配 Answer: 或 answer: 后紧跟一个范围为 A-J 的字母，并捕获该字母作为结果。
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)" # 文本中最后一个独立的大写字母（范围 A-J）
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(oai_client, inference_batch):
    start = time.time()
    response_batch = []
    pred_batch = []
    for prompt in tqdm(inference_batch, ncols=75):
        response = oai_client.call(prompt, return_logits=False, max_tokens=max_new_tokens, temperature=temperature)
        response_batch.append(response)
        if response is None:
            logging.info("response is None")
            pred_batch.append(None)
            continue
        pred = extract_answer(response)
        if not pred:
            logging.info("answer extract failed:\n" + response)
        pred_batch.append(pred)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    return pred_batch, response_batch, inference_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, oai_client, val_df, test_df, output_path):
    global choices
    inference_batches = []

    logging.info("generating prompts for " + subject)
    for i in tqdm(range(len(test_df)), ncols=75):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            if crop_prompt(prompt, max_model_length - max_new_tokens) == prompt:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    logging.info("evaluating " + subject)
    pred_batch, response_batch, inference_batch = batch_inference(oai_client, inference_batches)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        curr["model_input"] = inference_batch[j]
        res.append(curr)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))
    return accu, corr, wrong


def main():
    oai_client = Openai(
        apis=API_INFOS[args.model]
    )
    # res = oai_client.call("hello")

    full_test_df, full_val_df = load_mmlu_pro()
    
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    # print("selected subjects:\n" + "\n".join(selected_subjects))
    
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count = eval_cot(subject, oai_client, val_df, test_df, output_path)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    # parser.add_argument("--data_path", "-dp", type=str, default="hf_hub") # TIGER-Lab/MMLU-Pro
    parser.add_argument("--data_path", "-dp", type=str, default="/home/shaohanh/qilongma/blob/datasets/MMLU-Pro/data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.9")
    parser.add_argument("--model", "-m", type=str, default="OpenAI-GPT-4o")
    args = parser.parse_args()

    global_record_file = args.global_record_file
    os.makedirs(args.save_dir, exist_ok=True)
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    os.makedirs(save_result_dir, exist_ok=True)
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()
