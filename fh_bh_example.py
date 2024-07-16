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

from datasets import Dataset

import json

from typing import List, Dict

sys.setrecursionlimit(1000000)


# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
script_args.output_dir = 'fp_output'
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

def Q_Deq_SymQ(input, num_bits, my_per_channel=False):

    if not my_per_channel:
        max_input = (
            torch.max(torch.abs(input), dim=-1, keepdim=True)[0]
            .expand_as(input)
            .detach()
        )
    else:

        max_input = (
            torch.max(torch.abs(input), dim=0, keepdim=True)[0]
            .expand_as(input)
            .detach()
        )

    s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)

    # q_output = torch.round(input * s)

    output = torch.round(input * s).div(s + 1e-6)

    return output


def client_eval_resample(g_model, client_train_dataset, forward_hook=False):

    # training_args.train_batch_size = 16

    # g_model.eval()

    if forward_hook:
        activations_dic, handles = register_activation_input_hooks(g_model, 0, 'lora')

    training_args.per_device_eval_batch_size = 8
    c_trainer = SFTTrainer(
        model=g_model,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=None,
        eval_dataset=client_train_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator

    )

    sample_loss = c_trainer.loss_per_sample_evaluate()

    if forward_hook:
        save_activations_to_safetensor(activations_dic, os.path.join(script_args.output_dir, f"activations-global-{round + 1}.pt"))
        for i in range(len(handles)):
            handles[i].remove()

        del activations_dic, handles

    del c_trainer

    torch.cuda.empty_cache()

    # Delete g_trainer

    return sample_loss

def get_top_percent_indices(lst, percent):

    sorted_lst_with_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)

    cutoff_index = int(len(lst) * percent)

    top_indices = [index for index, _ in sorted_lst_with_indices[:cutoff_index]]

    return top_indices

def token_compression(sample, instrut_ratio = 0.8, response_ratio=0.8):
    from llmlingua import PromptCompressor

    llm_lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,  # Whether to use llmlingua-2
        # llmlingua2_config={
        #     "max_batch_size": 100,
        #     "max_force_token": 4096,
        # }
    )

    compressed_instruction =  llm_lingua.compress_prompt(sample['instruction'], rate=instrut_ratio, force_tokens=['\n', '?'])
    compressed_response = llm_lingua.compress_prompt(sample['response'], rate=response_ratio, force_tokens=['\n', '?'])

    # ## Or use LLMLingua-2-small model
    # llm_lingua = PromptCompressor(
    #     model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    #     use_llmlingua2=True,  # Whether to use llmlingua-2
    # )

    del llm_lingua

    return {
        'instruction': compressed_instruction['compressed_prompt'],
        'response': compressed_response['compressed_prompt']
    }

def get_top_down_percent_indices(lst, percent):

    sorted_lst_with_indices = sorted(enumerate(lst), key=lambda x: x[1], reverse=True)

    cutoff_index = int(len(lst) * percent)

    t_idx = [index for index, _ in sorted_lst_with_indices[:cutoff_index]]
    b_idx = [index for index, _ in sorted_lst_with_indices[cutoff_index:]]

    return t_idx, b_idx


def global_eval_save(g_model, global_eval_dataset, forward_hook=False):

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

    g_trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}"))


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





hetero_local_dict_list = []

if script_args.use_peft:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # hetero_local_dict_list.append(copy.deepcopy(get_peft_model_state_dict(model)))
    # model.base_model.delete_adapter('default')

# heterogeneous lora modules list

# ===== Define the global and local models =====
# global_dict = copy.deepcopy(get_peft_model_state_dict(model))
if script_args.use_peft:
    global_dict = copy.deepcopy(get_peft_model_state_dict(model))

else:
    global_dict = copy.deepcopy(model.state_dict())




local_dict_list = [copy.deepcopy(global_dict) for i in range(fed_args.num_clients)]

proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)


# ===== Define the tokenizer =====
# tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False, padding_side="left")   # cannot be use in sft
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
from datasets import concatenate_datasets



# ===== Start federated training =====
training_loss = [[] for i in range(fed_args.num_clients)]
global_loss = []



local_compress_flag = [0 for i in range(fed_args.num_clients)]
rewrite_flag = [0 for i in range(fed_args.num_clients)]

client_round_loss_order = {client: [] for client in range(fed_args.num_clients)}



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
                                             script_args)  # randomly get the required sub-dataset for this round
        training_data = sub_dataset
        eval_data = None


        if script_args.rewrite:

            if local_compress_flag[client] != 1:
                print('Have not been rewrite')
                local_compress_flag[client] = 1

            # if data_downsample:
            loss_list = client_eval_resample(model, training_data, forward_hook=False)

            client_round_loss_order[client].append(loss_list)

            if script_args.rewrite_top:
                t_idx, b_idx = get_top_down_percent_indices(loss_list, script_args.rewrite_percent) # descending order
                w4w_data = training_data.select(t_idx)
                remain_data = training_data.select(b_idx)
            else:
                t_idx, b_idx = get_top_down_percent_indices(loss_list, 1-script_args.rewrite_percent)
                w4w_data = training_data.select(b_idx)
                remain_data = training_data.select(t_idx)

            if 'code' in script_args.dataset_name:
                rewrite_template = """ Below is an instruction that describes a task along with a reference answer. 
                Refer to the reference answer, write your own answer in code. Put your answer after "### Your response:" and only show the code part.
                ### Instruction: {}
                ### Reference Answer: {}
                ### Your response:
                """
            else:
                rewrite_template = """ Below is an instruction that describes a task along with a reference answer. 
                Refer to the reference answer, write your own answer. Put your answer after "### Your response:".
                ### Instruction: {}
                ### Reference Answer: {}
                ### Your response:
                """


            w4w_data_list = []
            current_data = []
            raw_rewrite_data = []
            response_data = []

            for i in range(len(w4w_data)):
                # temp_data = rewrite_template.format(training_data[i]['instruction'], training_data[i]['response'], tokenizer.eos_token)
                temp_data = rewrite_template.format(w4w_data[i]['instruction'], w4w_data[i]['response'])
                response_data.append(w4w_data[i]['response'])
                current_data.append(temp_data)
                w4w_data_list.append(w4w_data[i])

            temp_tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=False,
                                                      padding_side="left")
            if temp_tokenizer.pad_token is None:
                temp_tokenizer.pad_token = temp_tokenizer.unk_token

            print('Start Rewriting')


            bs = script_args.rewrite_bs  # avoid OOD
            for batch_start_idx in tqdm(range(0, len(current_data), bs)):
                if batch_start_idx+bs > len(current_data) - 1:
                    break
                batched_inputs = temp_tokenizer(current_data[batch_start_idx:batch_start_idx+bs],
                                           return_tensors='pt', padding=True, truncation=False).to(model.device)
                response_data_inputs = temp_tokenizer(response_data[batch_start_idx:batch_start_idx+bs],
                                           return_tensors='pt', padding=True, truncation=False).to(model.device)
                # m_l = int(scaler * batched_inputs['input_ids'].size()[1])
                m_n_l = int(script_args.rewrite_scaler * response_data_inputs['input_ids'].size()[1])
                batched_outputs = model.generate(batched_inputs.input_ids, max_new_tokens=m_n_l)
                batched_output_str = temp_tokenizer.batch_decode(batched_outputs[:,batched_inputs.data['input_ids'].size(1):], skip_special_tokens=True)
                raw_rewrite_data.extend(batched_output_str)
                # break

            for i in range(len(raw_rewrite_data)):
                w4w_data_list[i]['response'] = raw_rewrite_data[i]

            DATA = Dataset.from_list(w4w_data_list)

            # concatanate
            training_data = concatenate_datasets([DATA, remain_data])
            training_data.shuffle(seed=2024)

            del current_data, w4w_data_list, raw_rewrite_data

                # print(outputs_str)
            print('End Rewriting')



        compress_token=False
        if compress_token:
            if local_compress_flag[client] != 1:
                training_data = training_data.map(token_compression)
                print('Token Compress Done')
                local_compress_flag[client] = 1

        # data_downsample = False
        # general_data_mix = False
        # top_percent = 0.6
        # full_data_num = len(training_data)
        # if data_downsample:
        #     loss_list = client_eval_resample(model, training_data, forward_hook=False)
        #     idx = get_top_percent_indices(loss_list, top_percent)
        #     training_data = training_data.select(idx)
        #     print('Number of data', len(training_data))
        #
        # if general_data_mix:
        #
        #     global_general_dataset = get_local_dataset('alpaca-gpt4')
        #     global_general_dataset = modified_process_sft_general_dataset('alpaca-gpt4',
        #                                                                   global_general_dataset, num_sample = int(full_data_num - len(training_data)), seed=2024)
        #     training_data = concatenate_datasets([training_data, global_general_dataset])
        #     training_data.shuffle(seed = 2024)
        #
        #     print("Mixing is done")

        if script_args.BP_free:
            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate,
                                          1e-6)  # manually schedule the learning rate
        else:
            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.adam_learning_rate,
                                          1e-6)  # manually schedule the learning rate

        training_args = get_training_args(script_args, new_lr)


        if script_args.BP_free:
            # ===== Train local model on the client side =====
            trainer = get_fed_local_sft_trainer_fp(
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
                zo_eps=2e-4,
            )
        else:
            #BP
            trainer = get_fed_local_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_train_dataset=training_data,
                local_eval_dataset=eval_data,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=None,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=None,
                global_auxiliary=None,
            )



        results = trainer.train()
        training_loss[client].append(results.training_loss)

        # ===== Save local model =====
        # currently save the final global model for further evaluation
        # if (round + 1) % fed_args.num_rounds == 1:
        # trainer.save_model(os.path.join(script_args.output_dir, f"local-{client}-checkpoint-{round + 1}"))

        if script_args.use_loc_fh:
            # ===== Save results and  Operate handles =====
            save_activations_to_safetensor(activations_dic, os.path.join(script_args.output_dir, f"activations-{client}-{round + 1}.pt"))
            for i in range(len(handles)):
                handles[i].remove()
            del activations_dic, handles



        # ===== Client transmits local information to server =====
        if fed_args.fed_alg == 'scaffold':
            auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()



        local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))  # deep copy is needed!
        print('Done')

    if script_args.rewrite:
        file_name = 'client_losses_order_dict_{}.json'.format(script_args.rewrite_percent)
        if not os.path.isfile(file_name):
            with open(file_name, 'w') as file:
                json.dump(client_round_loss_order, file)

    # ===== Server aggregates the local models =====
    global_dict, global_auxiliary = global_aggregate(
        script_args , fed_args, global_dict, local_dict_list, sample_num_list, \
        clients_this_round, round, proxy_dict=proxy_dict, \
        opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict), overall_drop_rate=0.0
    )

    if (round + 1) % fed_args.vanilla_save_freq == 0:
        set_peft_model_state_dict(model, global_dict)

        g_trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=None,
            eval_dataset=None,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
        g_trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round + 1}"))
        del g_trainer

    if (round + 1) % fed_args.save_model_freq != 0 :

        if script_args.quantize:
            print('Download Quantization')
            for k, v in global_dict.items():
                if 'lora_A' in k:
                    v.data = Q_Deq_SymQ(v, num_bits=script_args.q_bit)
                else:
                    v.data = Q_Deq_SymQ(v, num_bits=script_args.q_bit)
            print('Global-to-Client SymQ and DeSymQ are Done')


    # set_peft_model_state_dict(model, global_dict)  # Update global model



    if (round + 1) % fed_args.save_model_freq == 0:
        # if script_args.use_glob_fh:
        #     glo_eval_loss = global_eval(model, global_general_dataset, forward_hook=True)   # eval on global general data

        set_peft_model_state_dict(model, global_dict)

        # only evaluate on 4000
        remain_dataset.select(range(4000))

        glo_eval_loss = global_eval_save(model, remain_dataset, forward_hook=False)
        print('global loss:', glo_eval_loss)
        global_loss.append(glo_eval_loss)


        if script_args.quantize:
            print('Download Quantization')
            for k, v in global_dict.items():
                if 'lora_A' in k:
                    v.data = Q_Deq_SymQ(v, num_bits=script_args.q_bit)
                else:
                    v.data = Q_Deq_SymQ(v, num_bits=script_args.q_bit)
            print('Global-to-Client SymQ and DeSymQ are Done')



    np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    np.save(os.path.join(script_args.output_dir, "global_loss.npy"), np.array(global_loss))





