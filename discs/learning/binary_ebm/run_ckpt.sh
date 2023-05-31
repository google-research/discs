#!/bin/bash

dataset=caltech
# model size
num_channels=64

data_dir=${HOME}/gs/discs/storage/data
pt_path=${HOME}/gs/discs/storage/models/binary_ebm/${dataset}-${num_channels}/best_ckpt-2299.pt
save_root=${HOME}/gs/discs/storage/models/binary_ebm/${dataset}-${num_channels}

export CUDA_VISIBLE_DEVICES=

python -m discs.learning.binary_ebm.pt2params \
  --config="discs/learning/binary_ebm/exp_config.py:${dataset}-${num_channels}" \
  --pt_path=$pt_path \
  --save_root=$save_root \
  $@
