#!/bin/bash

# experiment_folder=discs-ising-chian_length_sweep_56835483
# evaluation_type=ess
# key=chain_length

# experiment_folder=discs-bernoulli-shape_sweep_62671971
# evaluation_type=ess
# key=shape

# experiment_folder=discs-bernoulli-lbf_sweep_62672009
# evaluation_type=ess
# key=balancing_fn_type

# experiment_folder=discs-categorical-num_categories_sweep_62668345
# evaluation_type=ess
# key=num_categories

# experiment_folder=discs-fhmm-categ_sweep_62701406
# evaluation_type=ess
# key=name

# # Example of the CO experiment
# experiment_folder=discs-maxclique-rb_sampler_sweep_57644072
# evaluation_type=co
# key=name


# # Example of the LM experiment
# experiment_folder=discs-lm-base_bert_sampler_sweep_57902239
# evaluation_type=lm
# key=name

data_path="./discs/plot_results/${experiment_folder}"

python -m discs.plot_results.plot_results \
  --gcs_results_path=$data_path \
  --evaluation_type=$evaluation_type \
  --key=$key \
