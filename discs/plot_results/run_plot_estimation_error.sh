#!/bin/bash

experiment_folder=discs-bernoulli-model_config_sweep_57481819


data_path="./discs/plot_results/${experiment_folder}"

python -m discs.plot_resultsplot_estimation_error \
  --gcs_results_path=$data_path \
