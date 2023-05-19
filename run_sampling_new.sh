#!/bin/bash
sgen_type=$1
quant_type=$2
dataset=$3
w_bit=$4
a_bit=$5
is_dynamic=$6
if [[ "${is_dynamic}" == "True" ]]; then
    dynamic_cmd="--use_dynamic"
else
    dynamic_cmd="\\"
fi

ckpt_path=$7
epoch=$8
batch_size=$9

echo ==================
echo ${sgen_type}
echo ${quant_type}
echo ${dataset}
echo ${w_bit}
echo ${a_bit}
echo ${dynamic_cmd}
echo ${ckpt_path}
echo ${epoch}
echo ==================
N_SAMPLES=50000

# gpus=("$@")
gpus=(0 1 2 3 4 5 6 7) #please modify it manually
n_gpu=${#gpus[@]}
n_sample_per_process=$(($N_SAMPLES / $n_gpu))

echo "Using GPU : ["${gpus[@]}"]"
echo "Total Sample : "$n_sample_per_process" X "$n_gpu" = "$(($n_sample_per_process * $n_gpu))

echo "Epoch: "$epoch" checkpoint" 

echo "Sampling..."
logdir=/SSD/stable_diffusion/QAT/samples/${dataset}_W${w_bit}A${a_bit}_${sgen_type}_${quant_type}_${is_dynamic}
for i in ${!gpus[@]}
do
    n_start=$(($i * $n_sample_per_process))

    SGEN_TYPE=${sgen_type} QUANT_TYPE=${quant_type} python sample_diffusion_dist.py \
        --ckpt ${ckpt_path}/${epoch}\
        --config /NAS/LJW/LSQ_diffusion/stable-diffusion/${dataset}_sampling.yaml  \
        --logdir  $logdir \
        --n_samples $n_sample_per_process \
        --n_start $n_start \
        --custom_steps 200 \
        --gpus ${gpus[$i]} \
        --batch_size 512 \
        --n_bit_w ${w_bit} \
        --n_bit_a ${a_bit} \
        --name epoch_${epoch}\
        ${dynamic_cmd} \
        &> asd.log &
done
wait

echo "Done!"
touch ${logdir}/DONE.txt
#
############################################

# Calculate Fid & Inception Score
python scripts/evaluate_fid.py \
    --logdir $logdir \
    --gpu ${gpus[0]} \
    --dataset ${dataset} \
    --name epoch_${epoch}\
