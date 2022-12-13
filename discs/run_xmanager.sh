#!/bin/bash

# model=bernoulli
# sampler=randomwalk
# xmanager launch discs/xm_launcher.py -- \
#     --model_config="discs/models/configs/${model?}_config.py" \
#     --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
#     --xm_resource_alloc="user:xcloud/${USER}" \
#     --experiment_name="Sampling_Experiment_${sampler?}-${model?}" \

models="bernoulli ising"
samplers="randomwalk gibbswithgard"

for model in $models
do
for sampler in $samplers
do
    echo "running $sampler on $model"
    xmanager launch discs/xm_launcher.py -- \
    --model_config="discs/models/configs/${model?}_config.py" \
    --sampler_config="discs/samplers/configs/${sampler?}_config.py" \
    --xm_resource_alloc="user:xcloud/${USER}" \
    --experiment_name="Sampling_Experiment_${sampler?}-${model?}" \

done
done
