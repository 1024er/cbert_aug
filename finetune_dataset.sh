set -x

export CUDA_VISIBLE_DEVICES=$1
python -u finetune_dataset.py --task_name $2

