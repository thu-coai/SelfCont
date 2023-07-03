import torch
import sys
import os
import numpy as np
from transformers import GPT2Tokenizer
from modeling_gpt2 import GPT2LMHeadModel
import traceback

model_name_path = sys.argv[1]
file_out = sys.argv[2]
device = sys.argv[3]
batch_size = int(sys.argv[4])
task_name = sys.argv[5]
data_ipt_file = sys.argv[6]

print("using %s"%device)
with open(data_ipt_file, "r") as fin:
    ipt = [line.strip() for line in fin]
tokenizer = GPT2Tokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
print(pad_token_id, eos_token_id)

model = GPT2LMHeadModel.from_pretrained(model_name_path, return_dict=True).to(device)

def pro(token_list, tokenizer):
    string = tokenizer.decode(token_list, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    string = string[:string.find("<|endoftext|>")].strip()
    return string

print("write to %s"%file_out)
with open(file_out, "w", encoding='utf-8') as fout:
    st, ed = 0, 0
    all_loss = []
    with torch.no_grad():
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            inputs = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs.input_ids[:, :50].to(device)
            attention_mask = inputs.attention_mask[:, :50].to(device)
            gen = model.generate(input_ids,
                    attention_mask=attention_mask,
                    do_sample=True, 
                    top_p=float(sys.argv[7]), 
                    top_k=tokenizer.vocab_size,
                    num_beams=1,
                    # do_sample=False,
                    # num_beams=4,
                    # num_beam_groups=10,
                    # num_return_sequences=10,
                    # diversity_penalty=10.,
                    # decoder_start_token_id=0, 
                    # decoder_input_ids=decoder_input_ids.input_ids.to(device)[:,:3],
                    max_length=512, 
                    early_stopping=False, 
                    output_scores=True,
                    return_dict_in_generate=True)            
            for ip, op in zip(ipt[st:ed], gen["sequences"]):
                decode_op = pro(op, tokenizer)
                print(pro(tokenizer.encode(ip), tokenizer))
                print(decode_op)
                print("="*10)
                fout.write(decode_op+"\n")
