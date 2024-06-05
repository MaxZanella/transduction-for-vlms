#!/bin/bash

DATA=$1
PROTO=$2
ARCH=$3
METHOD=$4
SHOT=$5

datasets=("imagenet" "sun397" "fgvc" "eurosat" "stanford_cars" "food101" "oxford_pets" "oxford_flowers" "caltech101" "dtd" "ucf101")
seeds=("1" "2" "3")

for dataset in "${datasets[@]}"; do
  for seed in "${seeds[@]}"; do
    python3 main.py --root_path "${DATA}" \
                    --dataset "$dataset" \
                    --method On-top \
                    --backbone "${ARCH}" \
                    --seed "$seed" \
                    --setting Few_shot \
                    --prototypes_path "${PROTO}" \
                    --prototypes_method "${METHOD}" \
                    --prototypes_dataset imagenet \
                    --prototypes_shots "${SHOT}"
  done
done