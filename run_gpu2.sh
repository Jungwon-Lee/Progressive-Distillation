#!/bin/sh

CUDA_VISIBLE_DEVICES=$1 python main.py \
    --base cifar10-ddpm.yaml \
    --train True \
    --scale_lr False \
    --name cifar-10_x0_predict_4096 \