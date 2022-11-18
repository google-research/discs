#!/bin/bash


export XLA_FLAGS='--xla_force_host_platform_device_count=4'

python -m discs.experiments.main_sampling \
  --model_config="models/configs/${model?}_config.py" \
  --sampler_config="samplers/configs/${sampler?}_config.py" \
