# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os

from tqdm import tqdm

print("Loading tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained('../../llm_model/llama2-7b')
tokenizer = AutoTokenizer.from_pretrained('../../llm_model/llama2-7b',
                                          use_fast=False, padding_side="right")  # why does padding right say wrong generated data?
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token
print("Tokenizer loaded!")


print("Loading model")
# model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

model = AutoModelForCausalLM.from_pretrained(
    '../../llm_model/llama2-7b',
    # quantization_config=quantization_config,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
# model = AutoModelForCausalLM.from_pretrained('../../llm_model/llama2-7b')
model = model.to('cuda:0')
print("Model loaded!")

n_vocab = 500 # number of initial tokens for synthesizing data on each GPU.

# i_start = sys.argv[1]
i_start = 2
if os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"):
    with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
        lines = f.readlines()
        inner_loop = len(lines) % n_vocab
        outer_loop = len(lines) // n_vocab
else:
    inner_loop = 0
    outer_loop = 0

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")

for j in tqdm(range(3 + outer_loop, 6)):
    for i in range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab):
        print(i)
        input_ids = torch.tensor([[i]]).cuda()
        print("generating")
        outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
        outputs = model.generate(outputs1, do_sample=True, max_length=512)
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        text_dict = {"text" : gen_text[0]}
        print(gen_text[0])
        with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "a") as f:
            f.write(json.dumps(text_dict, ensure_ascii=False))
            f.write('\n')