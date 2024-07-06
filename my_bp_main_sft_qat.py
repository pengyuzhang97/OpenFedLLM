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
from config import get_config, save_config, get_model_config, get_training_args

import torch

from trl import SFTTrainer

from collections import OrderedDict

# from prune import generator
from prune.pruners import *

import matplotlib.pyplot as plt


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the local dataset =====
dataset = get_local_dataset(script_args.dataset_name)
dataset, remain_dataset = modified_process_sft_data(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Load the global general dataset =====
global_general_dataset = get_local_dataset('alpaca-gpt4')
global_general_dataset = modified_process_sft_general_dataset('alpaca-gpt4', global_general_dataset, num_sample=100)




# # ===== Load the dataset =====
# dataset = get_dataset(script_args.dataset_name, script_args.local_data_dir)
# dataset, remain_dataset = process_sft_dataset(script_args.dataset_name, dataset, script_args.dataset_sample)

# ===== Split the dataset into clients =====
local_datasets = split_dataset(fed_args, script_args, dataset)
sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]


# # ===== Define the tokenizer =====
# tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.unk_token  # following vicuna
#
# # ===== Define the formatting function (cater to TRL SFTTrainer)=====
# formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
# response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
#                         2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
# data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)







def get_score(sft_instance, model):

    train_dataloader = sft_instance.get_train_dataloader()

    scorer = IterSNIP_no_mask(model)

    score_dict = scorer.score_llm_wo_mask(sft_instance, model, train_dataloader)

    return score_dict





def global_eval(g_model, global_eval_dataset, forward_hook=False):

    # training_args.train_batch_size = 16

    # g_model.eval()

    if forward_hook:
        activations_dic, handles = register_activation_input_hooks(g_model, 0, 'lora')

    g_trainer = SFTTrainer(
        model=g_model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=None,
        eval_dataset=global_eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    glo_eval_result = g_trainer.evaluate()

    if forward_hook:
        save_activations_to_safetensor(activations_dic, os.path.join(script_args.output_dir, f"activations-global-{round + 1}.pt"))
        for i in range(len(handles)):
            handles[i].remove()

        del activations_dic, handles

    del g_trainer

    torch.cuda.empty_cache()

    # Delete g_trainer

    return glo_eval_result['eval_loss']


def register_activation_input_hooks(model, layer_index, lora_name):
    activation_dic = {}
    handles = []

    def get_activation_hook(name):
        def hook(model, input, output):
            # input is a batched tensor, I just want the first one
            activation_dic[name] = input[0].detach()
            # activation_dic[name] = input[0].detach()
        return hook

    for name, module in model.named_modules():
        if lora_name in name  and 'default' in name and 'layers.{}'.format(layer_index) in name and 'dropout' not in name:
            print(name)
            handles.append(  module.register_forward_hook(get_activation_hook(name)) )

    return activation_dic, handles


def save_activations_to_safetensor(activation_dic, file_name):
    torch.save(activation_dic, file_name)


def non_lsq_get_peft_model_state_dict(model):
    inter_dict = copy.deepcopy(get_peft_model_state_dict(model))
    dict = OrderedDict()
    for k, v in inter_dict.items():
        if 'quan' not in k:
            dict[k] = v

    del inter_dict

    return dict


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

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()




# ===== Define the global and local models =====
# global_dict = copy.deepcopy(get_peft_model_state_dict(model))
if fed_args.use_lsq:
    global_dict = non_lsq_get_peft_model_state_dict(model)   # deepcopy is inside
else:
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))



local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]
proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token  # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[
                        2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)



# """
# test global moder eval
# """



# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
global_loss = []
if fed_args.per_tuning:
    avg_local_eval_loss = []

for round in tqdm(range(fed_args.num_rounds)):

    clients_this_round = get_clients_this_round(fed_args, round)

    # eval_loss = [[] for i in range(len(clients_this_round))]

    eval_loss = []

    print(f">> ==================== Round {round + 1} : {clients_this_round} ====================")

    for client in range(fed_args.num_clients):



        if client not in clients_this_round:
            training_loss[client].append(-1)  # -1 is an indicator of not training
            continue

        set_peft_model_state_dict(model, global_dict)  # sync the global model to the local model


        if script_args.use_loc_fh:
            activations_dic, handles = register_activation_input_hooks(model, 0, 'lora')

        sub_dataset = get_dataset_this_round(local_datasets[client], round, fed_args,
                                             script_args)  # get the required sub-dataset for this round

        if fed_args.per_tuning:
            eval_size =  int(fed_args.per_eval_ratio * sub_dataset.num_rows )
            split_data = sub_dataset.train_test_split(test_size=eval_size, shuffle=False, seed=42)

            training_data = split_data['train']
            eval_data = split_data['test']

        else:
            training_data = sub_dataset
            eval_data = None


        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate,
                                      1e-6)  # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)

        # ===== Train local model on the client side =====
        trainer = get_fed_local_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_train_dataset=training_data,
            local_eval_dataset= eval_data,
            formatting_prompts_func=formatting_prompts_func,
            data_collator=data_collator,
            global_dict=global_dict,
            fed_args=fed_args,
            script_args=script_args,
            local_auxiliary=auxiliary_model_list[client],
            global_auxiliary=global_auxiliary,
        )

        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Save local model =====
        # currently save the final global model for further evaluation
        # if (round + 1) % fed_args.num_rounds == 1:
        trainer.save_model(os.path.join(script_args.output_dir, f"local-{client}-checkpoint-{round + 1}"))

        score_dict = get_score(trainer, model)



        # store the tensors of the values in norm_score_dict in a list. Then rank the list and save the rank in a new list
        score_list = []

        score_list  = torch.cat([torch.flatten(v.cpu()) for  k, v in score_dict.items()])

        # print(score_list)

        sorted_score_list = sorted(score_list, reverse=False)


        # histgram of the scores
        plt.figure()
        plt.hist(score_list, bins=100)
        plt.show()


        top_10_percent_score = sorted_score_list[int(len(sorted_score_list) * 0.1)]

        top_10_index = torch.tensor(score_list) > top_10_percent_score
        top_10_num_ratio = torch.sum(torch.tensor(score_list) > top_10_percent_score) / len(score_list)


        # print(top_10_percent_score)

        # I need to know the number of top 5% scores

        top_5_percent_score = sorted_score_list[int(len(sorted_score_list) * 0.05)]

        # print(top_5_percent_score)


        top_1_percent_score = sorted_score_list[int(len(sorted_score_list) * 0.01)]

        # I need to know the number of top 1% scores
        # print(top_1_percent_score)

        # I need to know the number of top 0.1% scores

        top_0_1_percent_score = sorted_score_list[int(len(sorted_score_list) * 0.001)]

        # print(top_0_1_percent_score)


        #
        # print(sorted_score_list)

        # print(score_dict)

        # print(score_list)



        if script_args.use_loc_fh:
            # ===== Save results and  Operate handles =====
            save_activations_to_safetensor(activations_dic, os.path.join(script_args.output_dir, f"activations-{client}-{round + 1}.pt"))
            for i in range(len(handles)):
                handles[i].remove()
            del activations_dic, handles

        if fed_args.per_tuning:
            eval_result = trainer.evaluate()
            eval_loss.append(eval_result['eval_loss'])



        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()


        if fed_args.use_lsq:
            local_dict_list[client] = non_lsq_get_peft_model_state_dict(model)
        else:
            local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))  # deep copy is needed!

        print('Done')

    # avg of local loss for this current round and save
    if fed_args.per_tuning:
        avg_local_eval_loss.append(np.mean(eval_loss))


    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict), overall_drop_rate=0.0
    )



    set_peft_model_state_dict(model, global_dict)  # Update global model

    # glo_eval_loss = global_eval(model, remain_dataset)

    if script_args.use_glob_fh:
        glo_eval_loss = global_eval(model, global_general_dataset, forward_hook=True)

    # global_loss.append(glo_eval_loss)
    # print('Global Loss in Round{} is {}'.format(round,glo_eval_loss))


    # ===== Save the global model =====
    # currently save the final global model for further evaluation
    # if (round + 1) % fed_args.num_rounds == 1:
    trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}"))

    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    np.save(os.path.join(script_args.output_dir, "global_loss.npy"), np.array(global_loss))
    if fed_args.per_tuning:
        np.save(os.path.join(script_args.output_dir, "avg_local_eval_loss.npy"), np.array(avg_local_eval_loss))




