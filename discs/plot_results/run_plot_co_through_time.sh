#!/bin/bash

experiment_folder=discs-mis-er_010_lambda_58250366
graphtype=mis

data_path="./discs/plot_results/${experiment_folder}"

python -m discs.plot_results.plot_co_through_time \
  --gcs_results_path=$data_path \
  --graphtype=$graphtype \
