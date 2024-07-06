import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *

from utils.process_dataset import get_local_dataset, modified_process_sft_data, modified_process_sft_general_dataset

from federated_learning import *
from fp_config import get_config, save_config, get_model_config, get_training_args

# ===== ZO function =====
from federated_learning.fed_local_sft_fp import get_fed_local_sft_trainer_fp

import torch

from trl import SFTTrainer

from collections import OrderedDict

import sys


sys.setrecursionlimit(1000000)


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
script_args.output_dir = 'memory_output'
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the local dataset =====
dataset = get_local_dataset(script_args.dataset_name)
dataset, remain_dataset = modified_process_sft_data(script_args.dataset_name, dataset, script_args.dataset_sample)

# # ===== Load the global general dataset =====
# global_general_dataset = get_local_dataset('alpaca-gpt4')
# global_general_dataset = modified_process_sft_general_dataset('alpaca-gpt4', global_general_dataset, num_sample=100)




# # ===== Load the dataset =====
# dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
# dataset, remain_dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]



# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

if script_args.load_in_8bit or script_args.load_in_4bit:
    #         1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
    #         head to fp32
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
if script_args.use_peft:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()



# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token  # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
# response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
#                         2:] # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2

response_template_ids = tokenizer.encode(response_template)[
                        2:]


data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)



# """
# test global moder eval
# """



# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
global_loss = []
if fed_args.per_tuning:
    avg_local_eval_loss = []



sub_dataset = get_dataset_this_round(local_datasets[0], round, fed_args,
                                     script_args)  # randomly get the required sub-dataset for this round


training_data = sub_dataset
eval_data = None


training_args = get_training_args(script_args, script_args.learning_rate)

# # fp trainer.py
# trainer = get_fed_local_sft_trainer_fp(
#     model=model,
#     tokenizer=tokenizer,
#     training_args=training_args,
#     local_train_dataset=training_data,
#     local_eval_dataset=eval_data,
#     formatting_prompts_func=formatting_prompts_func,
#     data_collator=data_collator,
#     global_dict=None,
#     fed_args=fed_args,
#     script_args=script_args,
#     local_auxiliary=None,
#     global_auxiliary=None,
#     zo_eps=2e-4,
# )


# bp trainer.py
trainer = get_fed_local_sft_trainer(
    model=model,
    tokenizer=tokenizer,
    training_args=training_args,
    local_train_dataset=training_data,
    local_eval_dataset=eval_data,
    formatting_prompts_func=formatting_prompts_func,
    data_collator=data_collator,
    fed_args=fed_args,
    script_args=script_args,
    global_dict=None,
    local_auxiliary=None,
    global_auxiliary=None,

)




results = trainer.train()




