i!/bin/bash

models="categorical ising potts bernoulli"
samplers="dlmc path_auxiliary randomwalk gwg"
weight_fns="SQRT RATIO"

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
