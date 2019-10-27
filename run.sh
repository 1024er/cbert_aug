set -x
i=2
for sn in 1 2 3
do
  for st in 6 4
    do
        export CUDA_VISIBLE_DEVICES=$i 
        nohup python -u aug_dataset.py --sample_num $sn --sample_ratio $st --temp 1 --gpu $i >> log_aug_sn_${sn}_st_${st}_gpu_${i}_temp_1.log 2>&1 &
        let i=i+1 
    done
done

i=2
for sn in 1 2 3
do
  for st in 6 4
    do
        export CUDA_VISIBLE_DEVICES=$i
        nohup python -u aug_dataset.py --sample_num $sn --sample_ratio $st --temp 0.5 --gpu $i >> log_aug_sn_${sn}_st_${st}_gpu_${i}_temp_0.5.log 2>&1 &
        let i=i+1
    done
done
