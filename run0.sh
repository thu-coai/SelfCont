data_name=wp #wikitext or wp
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