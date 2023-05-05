#!/bin/bash

models="text_infilling" #maxclique mis maxcut categorical rbm potts ising bernoulli"
samplers="path_auxiliary" #dmala path_auxiliary dlmc randomwalk gwg"

for model in $models
do
for sampler in $samplers
do
    echo "running $sampler on $model"
    model=$model sampler=$sampler ./discs/experiment/run_sampling_local.sh 
   	 
done
done
