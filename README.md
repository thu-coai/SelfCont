# SelfContrast
Code for the paper [Mitigating the Learning Bias towards Repetition by Self-Contrastive Training for Open-Ended Generation](https://arxiv.org/abs/2307.01542) (ACL 2023 Short Findings paper)



## Prerequisites

The code is written in PyTorch library. Main dependencies are as follows:

- Python: 3.6.9
- torch: 1.8.1
- transformers: 4.6.1

Other dependencies can be found in `requirements.txt`



## Computing infrastructure

We train HINT based on the platform: 

- OS: Ubuntu 16.04.3 LTS (GNU/Linux 4.4.0-98-generic x86_64)
- CUDA Version: 10.1
- GPU: NVIDIA Tesla V100



## Quick Start

#### 1. Datasets

The full data can be downloaded from [THUcloud](https://cloud.tsinghua.edu.cn/f/e18739f2e4944b48aefb/?dl=1).

#### 2. Training SelfCont

The initial checkpoint of GPT2 can be downloaded from [HuggingFace](https://huggingface.co/gpt2). We provide our checkpoints on [THUcloud](https://cloud.tsinghua.edu.cn/f/146540e0843c441f99e5/?dl=1).

- The 1st stage (get the premature checkpoint): Execute the following command (or run `bash ./run0.sh` directly): 

  ```shell
  data_name=wikitext
  env CUDA_VISIBLE_DEVICES=0 python3 -u ./run_clm.py \
    --model_name_or_path gpt2 \
    --train_file ./${data_name}_data/train.txt \
    --validation_file ./${data_name}_data/val.txt \
    --do_train \
    --do_eval \
    --num_train_epochs 100 \
    --max_eval_samples 1000 \
    --dataloader_num_workers 64 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --output_dir ./${data_name}_f0_ckpt \
    --logging_steps 5 \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --cache ../cache
  ```

  The 1st training stage is exactly the same as fine-tuning the standard GPT2 model.

- The 2nd stage (get the final model): Execute the following command (or run `bash ./run.sh` directly): 

  ```shell
  data_name=wikitext
  env CUDA_VISIBLE_DEVICES=0 python3 -u ./run_clm.py \
    --model_name_or_path ./${data_name}_f0_ckpt \
    --model_name_or_path2 ./${data_name}_f0_ckpt \
    --train_file ./${data_name}_data/train.txt \
    --validation_file ./${data_name}_data/val.txt \
    --do_train \
    --do_eval \
    --num_train_epochs 100 \
    --max_eval_samples 1000 \
    --dataloader_num_workers 64 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --output_dir ./${data_name}_selfcont_ckpt \
    --logging_steps 5 \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --cache ../cache
  ```
  

####  3. Inference

Execute the following command to generate texts (or run `bash ./gen.sh` directly): 

```shell
data_name=wikitext
model_ckpt_path=./${data_name}_selfcont_ckpt
result_file=./result.txt
device=cuda:0
bsz=16	# batch size
task_name=test
data_ipt_file=./${data_name}_data/${task_name}.txt
topp=0	# p of top-p sampling, 0 means greedy decoding
python3 ./gen.py $model_ckpt_path $result_file $device $bsz $task_name $data_ipt_file $topp
```



#### 4. Evaluation

Execute the following command for evaluation: 

```shell
cd ./eval
python3 ./eval.py
```

You can change `result_list` in the script `eval.py` to specify the results you want to evaluate.



## Citation

Please kindly cite our paper if this paper and it is helpful.

```
@misc{guan2023mitigating,
      title={Mitigating the Learning Bias towards Repetition by Self-Contrastive Training for Open-Ended Generation}, 
      author={Jian Guan and Minlie Huang},
      year={2023},
      eprint={2307.01542},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
