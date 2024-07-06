import datasets
from datasets import load_dataset
import pandas as pd
from .conversation import get_conv_template
from functools import partial

def get_dataset(dataset_name, local_data_dir=None):

    if dataset_name in ["gsm8k"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="main")
    elif dataset_name in ["lighteval/MATH"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="all")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train_sft")
    else:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")

    return dataset


def get_local_dataset(data_name):
    if data_name in "lucasmccabe-lmi/CodeAlpaca-20k":
        # dataset = load_dataset('../data/llm/CodeAlpaca-20k/data', split='train')
        dataset = load_dataset('parquet',
                               data_files='../data/llm/CodeAlpaca-20k/data/train-00000-of-00001-e270777bb989ac86.parquet', split='train')

    if data_name in "FinGPT/fingpt-sentiment-train":
        dataset = load_dataset('parquet',
                               data_files='../data/llm/fingpt-sentiment-train/data/train-00000-of-00001-dabab110260ac909.parquet',
                               split='train')

    if data_name in 'medalpaca/medical_meadow_medical_flashcards':
        dataset = load_dataset('json',
                               data_files='../data/llm/medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json',
                               split='train')
    if data_name in 'alpaca-gpt4':
        dataset = load_dataset('parquet',
                                  data_files='../data/llm/alpaca-gpt4/train-00000-of-00001-6ef3991c06080e14.parquet',
                                  split='train')

    if data_name in 'TIGER-Lab/MathInstruct':
        dataset = load_dataset('json',
                               data_files='../data/llm/MathInstruct/MathInstruct.json',
                               split='train')

    # if data_name in ''

    return dataset



def modified_process_sft_data(dataset_name, dataset, dataset_sample):
    if dataset_name in "lucasmccabe-lmi/CodeAlpaca-20k" or dataset_name in "FinGPT/fingpt-sentiment-train":
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'],
                              desc=f"Preprocessing {dataset_name} for unified format.")

    elif dataset_name in 'medalpaca/medical_meadow_medical_flashcards':       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")

    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])

    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        if dataset_name in "lucasmccabe-lmi/CodeAlpaca-20k":
            num_sample = 16000  # 16000 for CodeAlpace global evaulation
        train_dataset = dataset.select(range(num_sample))
        remaining_dataset = dataset.select(range(num_sample, len(dataset)))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(train_dataset)} examples, Remaining_Dataset has {len(remaining_dataset)} =====")
    return train_dataset, remaining_dataset


def modified_process_sft_general_dataset(dataset_name, dataset, num_sample=None, seed=2024):
    if dataset_name in 'alpaca-gpt4':
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'],
                              desc=f"Preprocessing {dataset_name} for unified format.")
    dataset = dataset.shuffle(seed=seed)

    if num_sample is not None:
        num_sample = min(len(dataset), num_sample)
        train_dataset = dataset.select(range(num_sample))
        remaining_dataset = dataset.select(range(num_sample, len(dataset)))
        print(f">> ===== After processing, Dataset {dataset_name} has {len(train_dataset)} examples, Remaining_Dataset has {len(remaining_dataset)} =====")

    return train_dataset


def process_sft_dataset(dataset_name, dataset, dataset_sample):
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'],
                              desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["lighteval/MATH"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['level', 'type'])
    elif dataset_name in ['gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        train_dataset = dataset.select(range(num_sample))
        remaining_dataset = dataset.select(range(num_sample, len(dataset)))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(train_dataset)} examples, Remaining_Dataset has {len(remaining_dataset)} =====")
    return train_dataset, remaining_dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
    if dataset_name in ["Anthropic/hh-rlhf"]:
        dataset = dataset.map(partial(split_hh, template_name=template_name), load_from_cache_file=False)
    elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
        dataset = dataset.map(partial(split_ultrafeedback, template_name=template_name), load_from_cache_file=False)
        dataset = dataset.remove_columns(['prompt_id', 'messages', 'score_chosen', 'score_rejected'])
    
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    print(f">> ===== Data Example =====")
    print(dataset[0])
    print(f">> {'='*50}")
    return dataset
    
def find_common_prefix(str1, str2):
    prefix = ""
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix

def split_ultrafeedback(example, template_name="vicuna_v1.1"):
    conv_template = get_conv_template(template_name)

    conv_template.append_message(conv_template.roles[0], example["prompt"])
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = " " + example["chosen"][1]["content"]       # There might need a space in the front.
    example["rejected"] = " " + example["rejected"][1]["content"]
    return example

def split_hh(example, template_name="vicuna_v1.1"):
    common_prefix = find_common_prefix(example["chosen"], example["rejected"])

    conv_template = get_conv_template(template_name)

    sentence = common_prefix
    human_prefix_len = len("\n\nHuman: ")
    assistant_prefix_len = len("\n\nAssistant: ")
    sentence = sentence[human_prefix_len:]
    turn = "user"
    while True:
        if turn == "user":
            index = sentence.find("\n\nAssistant: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[0], sentence[:index])
                turn = "assistant"
                sentence = sentence[index + assistant_prefix_len :]
        elif turn == "assistant":
            index = sentence.find("\n\nHuman: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[1], sentence[:index])
                turn = "user"
                sentence = sentence[index + human_prefix_len :]
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = example["chosen"][len(common_prefix) - 1 :]     # -1 to include the space in the front.
    example["rejected"] = example["rejected"][len(common_prefix) - 1 :]
    return example



# if __name__ == "__main__":
#     data_name = 'CodeAlpaca-20k'
#
#     data_set = get_local_dataset(data_name)
#
#     print('done')