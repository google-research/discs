#!/bin/bash

experiment_folder=discs-maxclique-rb_sampler_sweep_57644072
graphtype=maxclique
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
