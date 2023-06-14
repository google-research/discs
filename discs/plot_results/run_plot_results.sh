#!/bin/bash

# experiment_folder=discs-ising-chian_length_sweep_56835483
# evaluation_type=ess
# key=chain_length

# experiment_folder=discs-potts-shape_sweep_57372547
# evaluation_type=ess
# key=shape

# experiment_folder=discs-potts-lbf_sweep_57372892
# evaluation_type=ess
# key=balancing_fn_type

# experiment_folder=discs-potts-num_categories_sweep_57372863
# evaluation_type=ess
# key=num_categories


# experiment_folder=discs-ising-model_config_sweep_extended_57873057
# evaluation_type=ess
# key=name

experiment_folder=discs-mis-er_density_penalty_58279781
evaluation_type=co
key=penalty
graphtype=mis

data_path="./${experiment_folder}"

python -m plot_results \
  --gcs_results_path=$data_path \
  --evaluation_type=$evaluation_type \
  --key=$key \
  --graphtype=$graphtype \


