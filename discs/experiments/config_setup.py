"""Setting up the config values."""

import importlib
import pickle

from absl import logging
from discs.common import configs as common_configs
from discs.samplers.locallybalanced import LBWeightFn
import yaml


def update_sampler_cfg(config):
  if 'balancing_fn_type' in config.sampler.keys():
    if _WEIGHT_FN.value == 'RATIO':
      config.sampler['balancing_fn_type'] = LBWeightFn.RATIO
    elif _WEIGHT_FN.value == 'MAX':
      config.sampler['balancing_fn_type'] = LBWeightFn.MAX
    elif _WEIGHT_FN.value == 'MIN':
      config.sampler['balancing_fn_type'] = LBWeightFn.MIN
    else:
      config.sampler['balancing_fn_type'] = LBWeightFn.SQRT


def update_model_cfg(config):
  if config.model.get('data_path', None):
    path = config.model.data_path
    model = pickle.load(open(path + 'params.pkl', 'rb'))
    config.model.params = model['params']
    model_config = yaml.unsafe_load(open(path + 'config.yaml', 'r'))
    config.model.update(model_config.model)


def get_main_config(model_config, sampler_config):
  config = common_configs.get_config()
  config.sampler.update(sampler_config)
  update_sampler_cfg(config)
  config.model.update(model_config)
  logging.info(config)
  update_model_cfg(config)

  if config.model.get('cfg_str', None):
    config.experiment.evaluator = 'co_eval'
    co_exp_default_config = importlib.import_module(
        'discs.experiments.configs.co_experiment'
    )
    config.experiment.update(co_exp_default_config.get_co_default_config())
    graph_exp_config = importlib.import_module(
        'discs.experiments.configs.%s.%s'
        % (config.model.name, config.model.graph_type)
    )
    config.experiment.update(graph_exp_config.get_config())
  else:
    config.experiment.evaluator = 'ess_eval'

  return config
