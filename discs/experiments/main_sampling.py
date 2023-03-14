"""Main script for sampling based experiments."""
import importlib
import pdb
import pickle
import discs.common.experiment_saver as saver_mod

from absl import app
from absl import flags
from absl import logging
from discs.common import configs as common_configs
from discs.experiment import experiment as experiment_mod
from discs.experiments import config_setup
#from discs.experiments import co_setup
from discs.experiments import sampling
from ml_collections import config_flags
import pdb
import jax
import os
from clu import metric_writers
from clu.metric_writers.summary_writer import SummaryWriter

_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')
_SAVE_DIR = flags.DEFINE_string('save_dir', './discs/results', 'Saving Dir')
_WEIGHT_FN = flags.DEFINE_string('weight_fn', 'SQRT', 'Balancing FN TYPE')
FLAGS = flags.FLAGS


def get_save_dir(config):
  save_folder = config.model.get('save_dir_name', config.model.name)
  return _SAVE_DIR.value + '_' + save_folder

def setup_logging(config):
  """Setup logging and writer."""
  if jax.process_index() == 0:
    logging.info(config)
    logging.info('process count: %d', jax.process_count())
    logging.info('device count: %d', jax.device_count())
    logging.info('device/host: %d', jax.local_device_count())
  logdir = os.path.join(config.experiment.save_root, 'logs')
  fig_folder = os.path.join(logdir, 'figures')
  if jax.process_index() == 0:
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    if not os.path.exists(fig_folder):
      os.makedirs(fig_folder)
    if config.experiment.use_tqdm:
      writer = SummaryWriter(logdir)
    else:
      writer = metric_writers.create_default_writer(logdir)
  else:
    writer = None
  config.experiment.fig_folder = fig_folder
  with open(os.path.join(config.experiment.save_root, 'config.yaml'), 'w') as f:
    f.write(config.to_yaml())
  return writer


def main_(_):
  config = config_setup.get_main_config(_MODEL_CONFIG.value, _SAMPLER_CONFIG.value, _WEIGHT_FN.value)

  # sampler
  sampler_mod = importlib.import_module(
      'discs.samplers.%s' % config.sampler.name
  )
  sampler = sampler_mod.build_sampler(config)

  global_key = jax.random.PRNGKey(0)

  # model
  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  model = model_mod.build_model(config)

  logger = setup_logging(config)

  sampling.solve_dataset(config, global_key, logger, datagen, model, sampler)

def main(_):
  config = config_setup.get_main_config(
      _MODEL_CONFIG.value, _SAMPLER_CONFIG.value, _WEIGHT_FN.value
  )

  # sampler
  sampler_mod = importlib.import_module(
      'discs.samplers.%s' % config.sampler.name
  )
  sampler = sampler_mod.build_sampler(config)




  # model
  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  model = model_mod.build_model(config)

  # experiment
  experiment = experiment_mod.build_experiment(config)

  # evaluator
  evaluator_mod = importlib.import_module(
      'discs.evaluators.%s' % config.experiment.evaluator
  )
  evaluator = evaluator_mod.build_evaluator(config)

  # chain generation
  metrics, running_time, acc_ratio, hops = experiment.get_results(
      model, sampler, evaluator
  )

#  # saver
#  saver = saver_mod.build_saver(get_save_dir(config), config)
#  saver.save_results(acc_ratio, hops, metrics, running_time)


if __name__ == '__main__':
  app.run(main)
