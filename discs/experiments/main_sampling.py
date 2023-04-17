"""Main script for sampling based experiments."""
import importlib
import logging
import discs.common.experiment_saver as saver_mod
import discs.common.utils as utils

from absl import app
from absl import flags
from discs.common import configs as common_configs
from ml_collections import config_flags
import os
import pdb


FLAGS = flags.FLAGS
_EXPERIMENT_CONFIG = config_flags.DEFINE_config_file('experiment_config', None)
_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')
_RUN_LOCAL = flags.DEFINE_boolean('run_local', False, 'if runnng local')


def get_save_dir(config):
  if _RUN_LOCAL.value:
    save_folder = config.model.get('save_dir_name', config.model.name)
    save_root = config.experiment.save_root + '/' + save_folder
  else:
    save_root = config.experiment.save_root
  return save_root


def get_main_config():
  config = common_configs.get_config()
  config.sampler.update(_SAMPLER_CONFIG.value)
  config.model.update(_MODEL_CONFIG.value)

  if config.model.get('graph_type', None):
    config.experiment.update(_EXPERIMENT_CONFIG.value)
    co_exp_default_config = importlib.import_module(
        'discs.experiments.configs.co_experiment'
    )
    config.experiment.update(co_exp_default_config.get_co_default_config())
  return config


def main(_):
  config = get_main_config()
  utils.setup_logging(config)

  # model
  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  model = model_mod.build_model(config)

  # sampler
  sampler_mod = importlib.import_module(
      'discs.samplers.%s' % config.sampler.name
  )
  sampler = sampler_mod.build_sampler(config)

  # experiment
  experiment_mod = getattr(
      importlib.import_module('discs.experiment.experiment'),
      f'{config.experiment.name}',
  )
  experiment = experiment_mod(config)

  # evaluator
  evaluator_mod = importlib.import_module(
      'discs.evaluators.%s' % config.experiment.evaluator
  )
  evaluator = evaluator_mod.build_evaluator(config)

  # saver
  saver = saver_mod.build_saver(get_save_dir(config), config)

  # chain generation
  experiment.get_results(model, sampler, evaluator, saver)


if __name__ == '__main__':
  app.run(main)
