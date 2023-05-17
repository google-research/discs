#!/bin/bash

experiment_folder=discs-mis-ertest_10k_sampler_sweep_56757032
graphtype=mis
graphkey=name
graphtitle='Sampler'
graphlabel='sampler'

data_path="./${experiment_folder}"

python -m plot_co_through_time \
  --gcs_results_path=$data_path \
  --graphkey=$graphkey \
  --graphtype=$graphtype \
  --graphtitle=$graphtitle \
  --graphlabel=$graphlabel \
