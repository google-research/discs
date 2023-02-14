#!/bin/bash

models="categorical"
samplers="dlmc"
weight_fns="SQRT"

for model in $models
do
for sampler in $samplers
do
for weight_fn in $weight_fns
do
    echo "running $sampler on $model"
    model=$model sampler=$sampler weight_fn=$weight_fn ./discs/experiments/run_sampling_local.sh 
   
done   	 
done
done
