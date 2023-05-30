#!/bin/bash

experiment_folder=discs-normcut-nets_sampler_sweep_56834938

data_path="./${experiment_folder}"

python -m plot_results \
  --gcs_results_path=$data_path \
