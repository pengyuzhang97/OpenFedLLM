from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig

from transformers.utils.quantization_config import GPTQConfig

import os
import json
from accelerate import Accelerator
import torch
from datetime import datetime, timedelta
from typing import List



# Define and parse arguments.
@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=100, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=20, metadata={"help": "the number of clients"})    # 20 / 2
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy, iid or non-iid"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})



    vanilla_save_freq: Optional[int] = field(default=5, metadata={"help": "the frequency to save the model excludes global_eval"})

    save_model_freq: Optional[int] = field(default=10, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})


    # per_tuning: Optional[bool] = field(default=False, metadata={"help": "whether to initialize personalized tuning"})
    # per_eval_ratio: Optional[float] = field(default=0.2, metadata={"help": "the ratio of eval to training data"})
    # use_lsq: Optional[bool] = field(default=False, metadata={"help": "whether to use lsq"})



@dataclass
class ScriptArguments:
    # candidate_data = ['TIGER-Lab/MathInstruct', "vicgalle/alpaca-gpt4", 'lucasmccabe-lmi/CodeAlpaca-20k']
    # canddate_model = ['../llm_model/llama2-7b']
    # model_name_or_path: Optional[str] = field(default="../llm_model/llama2-7b", metadata={"help": "the model name"})，
    # "../llm_model/MiniCPM-2B-stf-bf16", "../llm_model/gemma-2b-it"    ../llm_model/llama2-7b-chat

    model_name_or_path: Optional[str] = field(default="../llm_model/llama2-7b-chat", metadata={"help": "the model name"})

    dataset_name: Optional[str] = field(
        default="lucasmccabe-lmi/CodeAlpaca-20k", metadata={"help": "the dataset name"}
    )
    local_data_dir: Optional[str] = field(default=None, metadata={"help": "the local data directory if you want to use downloaded data"})

    # lucasmccabe-lmi/CodeAlpaca-20k, "vicgalle/alpaca-gpt4", 'TIGER-Lab/MathInstruct',
    # 'FinGPT/fingpt-sentiment-train', 'medalpaca/medical_meadow_medical_flashcards'


    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})

    optimizer: Optional[str] = field(default="sgd", metadata={"help": "optimizer, default adamw_hf, optional sgd"})
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate, default 5e-5, fp uses 1 / sqrt(d)"})
    adam_learning_rate: Optional[float] = field(default=1.5e-4, metadata={"help": "the learning rate, default 5e-5, fp uses 1 / sqrt(d)"})
    # vicuna and a
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Max Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=16, metadata={"help": "the r parameter of the LoRA adapters"})   # 8 and 16 is seems to be better
    peft_lora_alpha: Optional[int] = field(default=32, metadata={"help": "the alpha parameter of the LoRA adapters"}) # vanilla is 16

    # target_modules: List = field(default_factory=lambda: ["q_proj","k_proj", "v_proj"])

    logging_steps: Optional[int] = field(default=100, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})   # token and use_auth_token cannot be used together
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    template: Optional[str] = field(default="alpaca", metadata={"help": "the template to use"})
    seed: Optional[int] = field(default=2024, metadata={"help": "the seed to use"})
    dpo_beta: Optional[float] = field(default=0.0, metadata={"help": "the beta parameter of DPO"})
    dataset_sample: Optional[int] = field(default=20000, metadata={"help": "the number of samples to use from the dataset"})

    # use local_forward handles, bool value
    use_loc_fh: Optional[bool] = field(default=False, metadata={"help": "whether to use forward handles"})
    # use global_forward handles, bool value, on General dataset
    use_glob_fh: Optional[bool] = field(default=False, metadata={"help": "whether to use forward handles"})

    quantize : Optional[bool] = field(default=False, metadata={"help": "whether to use quantization"})
    q_bit: Optional[int] = field(default=8, metadata={"help": "quantize to q bit, default 8"})

    rewrite : Optional[bool] = field(default=False, metadata={"help": "whether to rewrite"})
    rewrite_scaler: Optional[float] = field(default=0.5, metadata={"help": "new number of tokens"})
    rewrite_percent: Optional[float] = field(default=0.1, metadata={"help": "percent of the largest losses"})
    rewrite_top: Optional[bool] = field(default=False, metadata={"help": "rewrite the top losses"})
    rewrite_bs: Optional[int] = field(default=8, metadata={"help": "be careful of OOD problem"})


    BP_free: Optional[bool] = field(default=False, metadata={"help": "whether to rewrite"})


parser = HfArgumentParser((ScriptArguments, FedArguments))
script_args, fed_args = parser.parse_args_into_dataclasses()

# ===== Define the LoraConfig =====
# lora_target_modules = ["q_proj","k_proj", "v_proj"]
lora_target_modules = ["q_proj","k_proj", "v_proj", 'o_proj']
# lora_target_modules = ["q_proj"]
# lora_target_modules = ["q_proj", "k_proj", "v_proj", 'o_proj', 'gate_proj', 'up_proj', 'down_proj']   # where is the gate?

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # init_lora_weights= False


    )
else:
    peft_config = None


# loftq_config =
#
# peft_config = LoraConfig(
#     r=script_args.peft_lora_r,
#     lora_alpha=script_args.peft_lora_alpha,
#     target_modules=lora_target_modules,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     init_lora_weights='loftq'
#     loftq_config =
#
# )


def get_config():
    return script_args, fed_args, peft_config

# ===== Define the training arguments =====
def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size ,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim= script_args.optimizer,
        learning_rate=new_lr,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type="constant",
        # fp16=True,    # use fp16 somehow causes in
        # bf16=True
    )
    return training_args

def get_model_config(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    elif script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = 'auto'
        quantization_config = None
        # torch_dtype = None
        torch_dtype = torch.float16
    return device_map, quantization_config, torch_dtype

def get_model_config_GPTQ(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        gptq_quantization_config = GPTQConfig(bits=4)

        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    return device_map, gptq_quantization_config, torch_dtype

def save_config(script_args, fed_args):
    now_time = (datetime.now()).strftime("%Y%m%d%H%M%S")
    dataset_name_split = os.path.basename(script_args.dataset_name)
    output_dir = f"{script_args.output_dir}/{dataset_name_split}_{script_args.dataset_sample}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_r{script_args.peft_lora_r}a{script_args.peft_lora_alpha}_{now_time}"
    while True:
        if not os.path.exists(output_dir):
            # os.mkdir(output_dir)
            os.makedirs(output_dir)
            break
        else:
            now_time = (datetime.now() + timedelta(seconds=1)).strftime("%Y%m%d%H%M%S")
            output_dir = f"{script_args.output_dir}/{dataset_name_split}_{fed_args.split_strategy}_{fed_args.fed_alg}_c{fed_args.num_clients}s{fed_args.sample_clients}_i{script_args.max_steps}_b{script_args.batch_size}a{script_args.gradient_accumulation_steps}_l{script_args.seq_length}_{now_time}"

    script_args.output_dir = output_dir
    with open(os.path.join(script_args.output_dir, "args.json"), "w") as f:
        combined_dict = {
            "script_args": asdict(script_args),
            "fed_args": asdict(fed_args),
            # 'lora_modules': lora_target_modules
        }
        json.dump(combined_dict, f, indent=4)