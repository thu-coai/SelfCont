data_name=wp #wikitext or wp
model_ckpt_path=./${data_name}_selfcont_ckpt
result_file=./result.txt
device=cuda:0
bsz=16
task_name=test
data_ipt_file=./${data_name}_data/${task_name}.txt
topp=0
python3 ./gen.py $model_ckpt_path $result_file $device $bsz $task_name $data_ipt_file $topp