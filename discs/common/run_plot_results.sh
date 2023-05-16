#!/bin/bash

experiment_folder=discs-categorical-chain_length_sweep_56696574
evaluation_type=ess
key=chain_length


experiment_folder=discs-categorical-batch_size_sweep_56693366
evaluation_type=ess
key=batch_size


experiment_folder=discs-categorical-dimension_sweep_56696499
evaluation_type=ess
key=shape


experiment_folder=discs-categorical-lbf_sweep_56696533
evaluation_type=ess
key=name


experiment_folder=discs-categorical-num_categories_sweep_56696865
evaluation_type=ess
key=num_categories


experiment_folder=discs-categorical-chain_length_sweep_56733761
evaluation_type=ess
key=chain_length


experiment_folder=discs-categorical-dimension_sweep_56733710
evaluation_type=ess
key=shape


experiment_folder=discs-categorical-lbf_sweep_56733728
evaluation_type=ess
key=balancing_fn_type

experiment_folder=discs-categorical-num_categories_sweep_56734165
evaluation_type=ess
key=num_categories

data_path="./${experiment_folder}"

python -m plot_results \
  --gcs_results_path=$data_path \
  --evaluation_type=$evaluation_type \
  --key=$key \

