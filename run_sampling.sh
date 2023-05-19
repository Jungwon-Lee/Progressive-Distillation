#!/bin/bash

N_SAMPLES=30000

# gpus=("$@")
gpus=(0 1 2 3 4 5 6 7) #please modify it manually
n_gpu=${#gpus[@]}
n_sample_per_process=$(($N_SAMPLES / $n_gpu))

echo "Using GPU : ["${gpus[@]}"]"
echo "Total Sample : "$n_sample_per_process" X "$n_gpu" = "$(($n_sample_per_process * $n_gpu))

echo "Epoch: "$epoch" checkpoint" 

echo "Sampling..."
logdir=/SSD/stable_diffusion/QAT/samples/Progressive_Teacher_test

epoch=477

ckpt_path="/NAS/LJW/progressive/teacher_10.ckpt"

w_bit=0
a_bit=0
epoch=1024 

for i in ${!gpus[@]}
do
    n_start=$(($i * $n_sample_per_process))

    python sample_diffusion_dist.py \
        --ckpt ${ckpt_path}\
        --config /NAS/LJW/progressive/cifar10-ddpm-distill_1024.yaml  \
        --logdir  $logdir \
        --n_samples $n_sample_per_process \
        --n_start $n_start \
        --custom_steps 200 \
        --gpus ${gpus[$i]} \
        --batch_size 512 \
        --n_bit_w ${w_bit} \
        --n_bit_a ${a_bit} \
        --name epoch_${epoch}\
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
    --dataset CIFAR-10 \
    --name epoch_${epoch}\
