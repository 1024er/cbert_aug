set -x
i=2
for dataset in mpqa rt-polarity stsa.binary stsa.fine subj TREC
do
    nohup bash baselines.sh $i ${dataset} >> log_baseline_${dataset} &
    let i=i+1
done
