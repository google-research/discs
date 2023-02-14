#!/bin/bash

# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# save_root=${SCRIPT_DIR}/../../../storage/models/rbm
save_root=./discs/storage/models/rbm

export CUDA_VISIBLE_DEVICES=0,1
export XLA_FLAGS="--xla_force_host_platform_device_count=2"

python -m discs.learning.rbm.main_rbm_training \
  --config="discs/learning/rbm/exp_config.py" \
  --save_root=$save_root \
  $@
