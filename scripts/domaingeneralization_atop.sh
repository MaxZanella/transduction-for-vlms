#!/bin/bash

DATA=$1
PROTO=$2
ARCH=$3
METHOD=$4

datasets=("imagenet_a" "imagenet_v2" "imagenet_r" "imagenet_sketch")
seeds=("1" "2" "3")

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    python3 main.py --root_path "${DATA}" \
                    --dataset "$dataset" \
                    --method On-top \
                    --backbone "${ARCH}" \
                    --seed "$seed" \
                    --setting Domain-generalization \
                    --prototypes_path "${PROTO}" \
                    --prototypes_method "${METHOD}" \
                    --prototypes_dataset imagenet \
                    --prototypes_shots 16
  done
done