#!/bin/bash

DATA=$1
ARCH=$2
METH=$3
SHOT=$4

datasets=("imagenet" "sun397" "fgvc" "eurosat" "stanford_cars" "food101" "oxford_pets" "oxford_flowers" "caltech101" "dtd" "ucf101")
seeds=("1" "2" "3")

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    python3 main.py --root_path "${DATA}" \
                    --dataset "$dataset" \
                    --method "${METH}" \
                    --backbone "${ARCH}" \
                    --seed "$seed" \
                    --shots "${SHOT}"
  done
done