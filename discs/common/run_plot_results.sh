#!/bin/bash

experiment_folder=discs-categorical-batch_size_sweep_56693366
evaluation_type=ess
key=batch_size


data_path="./${experiment_folder}"

python -m plot_results \
  --gcs_results_path=$data_path \
  --evaluation_type=$evaluation_type \
  --key=$key \

