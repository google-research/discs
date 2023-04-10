#!/bin/bash

#TODO: Kati migrate this to xmanager script.
# models="mis"
# samplers="path_auxiliary"

# for model in $models
# do
# for sampler in $samplers
# do
    # echo "running $sampler on $model"
xmanager launch discs/xm_launcher.py -- \
--config=${config?} \
--xm_resource_alloc="user:xcloud/xcloud-shared-user" \

# done
# done

