#!/bin/bash

#TODO: Kati migrate this to xmanager script.
models="mis"
samplers="randomwalk"

for model in $models
do
for sampler in $samplers
do
    echo "running $sampler on $model"
    xmanager launch discs/xm_launcher.py -- \
    --model_config="discs/models/configs/${model?}_config.py" \
    --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
    --xm_resource_alloc="user:xcloud/xcloud-shared-user" \
    --experiment_name="Sampling_Experiment_${sampler?}-${model?}" \

done
done

# --config=${config?} \