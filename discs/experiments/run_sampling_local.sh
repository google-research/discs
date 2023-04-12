#!/bin/bash
export XLA_FLAGS='--xla_force_host_platform_device_count=4'

echo "$model"
echo "$sampler"

python -m discs.experiments.main_sampling \
  --model_config="discs/models/configs/${model?}_config.py" \
  --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
