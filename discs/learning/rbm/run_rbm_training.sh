#!/bin/bash

dataset=mnist
num_categories=2
num_hidden=200

save_root=./discs/storage/models/rbm

export CUDA_VISIBLE_DEVICES=0,1
export XLA_FLAGS="--xla_force_host_platform_device_count=2"

python -m discs.learning.rbm.main_rbm_training \
  --config="discs/learning/rbm/exp_config.py:${dataset}-${num_categories}-${num_hidden}" \
  --save_root=$save_root \
  $@
