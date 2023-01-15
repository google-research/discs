#!/bin/bash

models="bernoulli"
samplers="randomwalk gwg path_auxiliary dlmc"

for model in $models
do
for sampler in $samplers
do
    echo "running $sampler on $model"
    model=$model sampler=$sampler ./discs/experiments/run_sampling_local.sh 
    
done
done