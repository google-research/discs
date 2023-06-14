#!/bin/bash

experiment_folder=discs-mis-er_010_lambda_58250366
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
