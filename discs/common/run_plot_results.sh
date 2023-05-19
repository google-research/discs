#!/bin/bash

# experiment_folder=discs-ising-chian_length_sweep_56835483
# evaluation_type=ess
# key=chain_length

# experiment_folder=discs-bernoulli-bernoulli_dlmcf_test_56881684
# evaluation_type=ess
# key=shape

# experiment_folder=discs-bernoulli-lbf_sweep_56835977
# evaluation_type=ess
# key=balancing_fn_type

# experiment_folder=discs-categorical-num_categories_sweep_56828693
# evaluation_type=ess
# key=num_categories


experiment_folder=discs-rbm-sampler_sweep_56930501
evaluation_type=ess
key=name


# experiment_folder=discs-normcut-nets_sampler_sweep_56834938
# evaluation_type=co
# key=name
# graphtype=normcut


data_path="./${experiment_folder}"

python -m plot_results \
  --gcs_results_path=$data_path \
  --evaluation_type=$evaluation_type \
  --key=$key \
  --graphtype=$graphtype \


