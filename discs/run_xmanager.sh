#!/bin/bash

xmanager launch discs/xm_launcher.py -- \
--config=${config?} \
--xm_resource_alloc="user:xcloud/xcloud-shared-user" \

