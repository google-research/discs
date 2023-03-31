"""Main script for sampling based experiments."""
import importlib
import pdb
import pickle
import discs.common.experiment_saver as saver_mod

from absl import app
from absl import flags
from absl import logging
from discs.common import configs as common_configs
from discs.experiments import config_setup
# from discs.experiments import co_setup
from ml_collections import config_flags
import pdb
import jax
import os
from clu import metric_writers
from clu.metric_writers.summary_writer import SummaryWriter

_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')
_SAVE_DIR = flags.DEFINE_string('save_dir', './discs/results', 'Saving Dir')
FLAGS = flags.FLAGS


def get_save_dir(config):
  save_folder = config.model.get('save_dir_name', config.model.name)
  return _SAVE_DIR.value + '_' + save_folder


def main(_):
  config = config_setup.get_main_config(
      _MODEL_CONFIG.value, _SAMPLER_CONFIG.value
  )

  # model
  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  model = model_mod.build_model(config)

  # sampler
  sampler_mod = importlib.import_module(
      'discs.samplers.%s' % config.sampler.name
  )
  sampler = sampler_mod.build_sampler(config)

  # experiment
  experiment_mod = getattr(importlib.import_module('discs.experiment.experiment'), f'{config.experiment.name}')
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
