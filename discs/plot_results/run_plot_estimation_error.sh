#!/bin/bash

experiment_folder=discs-bernoulli-model_config_sweep_57481819


data_path="./${experiment_folder}"

python -m plot_estimation_error \
  --gcs_results_path=$data_path \
