max_steps=2
num_rounds=10
batch_size=2
gradient_accumulation_steps=8
seq_length=512
num_clients=20
sample_clients=2
#lora_r=32
#lora_alpha=64   # twice of lora_r

lora_r=32
lora_alpha=64   # twice of lora_r

lr=5e-5

#candidate_data = ['TIGER-Lab/MathInstruct', "vicgalle/alpaca-gpt4"]

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
#lucasmccabe-lmi/CodeAlpaca-20k, "vicgalle/alpaca-gpt4", 'TIGER-Lab/MathInstruct'

dataset_name='TIGER-Lab/MathInstruct'
dataset_sample=20000
#model_name_or_path="meta-llama/Llama-2-7b-hf"
model_name_or_path="../llm_model/llama2-7b"
output_dir=./output

gpu=0
fed_alg="fedavg"

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "alpaca" \