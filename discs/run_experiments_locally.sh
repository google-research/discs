#!/bin/bash

models="mis maxcut categorical rbm potts ising bernoulli"
samplers="randomwalk path_auxiliary dlmc gwg"

for model in $models
do
for sampler in $samplers
do
    echo "running $sampler on $model"
    model=$model sampler=$sampler ./discs/experiments/run_sampling_local.sh 
   	 
done
done
