#!/bin/bash

dataset=dynamic_mnist
# model size
num_channels=64

data_dir=${HOME}/gs/discs/storage/data
save_root=${HOME}/gs/discs/storage/models/binary_ebm

export CUDA_VISIBLE_DEVICES=

python -m discs.learning.binary_ebm.main_binary_training \
  --config="discs/learning/binary_ebm/exp_config.py:${dataset}-${num_channels}" \
  --config.experiment.data_dir=${data_dir} \
  --save_root=$save_root \
  $@
