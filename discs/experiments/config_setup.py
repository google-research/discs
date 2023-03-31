"""Setting up the config values."""

import importlib
import logging
import pickle
import flax


from discs.common import utils
from absl import logging
from discs.common import configs as common_configs

import yaml
import jax.numpy as jnp

import pdb

def update_experiment_cfg(config):
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
    config.experiment.log_every_steps = 1


def get_main_config(model_config, sampler_config):
  config = common_configs.get_config()
  config.sampler.update(sampler_config)
  config.model.update(model_config)
  logging.info(config)
  update_experiment_cfg(config)
  return config
