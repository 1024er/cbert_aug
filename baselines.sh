set -x
export CUDA_VISIBLE_DEVICES=$1
dataset=$2

for d in 0.1 0.2 0.3 0.4 0.5 0.6 
do 
    python -u train_text_classifier_original.py --dataset $dataset --model cnn --dropout $d >> logs_baselines/${dataset}/cnn/log_${d}.log &
    python -u train_text_classifier_original.py --dataset $dataset --model rnn --dropout $d >> logs_baselines/${dataset}/rnn/log_${d}.log
done
