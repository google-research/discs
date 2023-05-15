#!/bin/bash

experiment_folder=discs-maxclique-rb_56691955
graphtype=maxclique
graphkey=decay_rate
graphtitle='decay_rate'
graphlabel='decay_rate'

experiment_folder=discs-maxclique-rb_56691955
graphtype=maxclique
graphkey=chain_length
graphtitle='chain_length'
graphlabel='chain_length'


experiment_folder=discs-maxclique-rb_56691955
graphtype=maxclique
graphkey=init_temperature
graphtitle='init_temp'
graphlabel='init_temp'

data_path="./${experiment_folder}"

python -m plot_co_through_time \
  --gcs_results_path=$data_path \
  --graphkey=$graphkey \
  --graphtype=$graphtype \
  --graphtitle=$graphtitle \
  --graphlabel=$graphlabel \
