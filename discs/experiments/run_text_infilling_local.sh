#!/bin/bash
export XLA_FLAGS='--xla_force_host_platform_device_count=4'

echo "$model"
echo "$sampler"

CUDA_VISIBLE_DEVICES=7 python -W ignore -m  discs.experiments.main_text_infilling \
  --model_config="discs/models/configs/${model?}_config.py" \
  --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
 # --save_dir=${save_dir} \
