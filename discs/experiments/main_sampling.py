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
from discs.samplers.locallybalanced import LBWeightFn
from ml_collections import config_flags
import yaml

_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')
_SAVE_DIR = flags.DEFINE_string('save_dir', './discs/results', 'Saving Dir')
_WEIGHT_FN = flags.DEFINE_string('weight_fn', 'SQRT', 'Balancing FN TYPE')
FLAGS = flags.FLAGS


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
  if config.model.name == 'rbm':
    path = config.model.data_path
    model = pickle.load(open(path + 'params.pkl', 'rb'))
    config.model.params = model['params']
    model_c = yaml.unsafe_load(open(path + 'config.yaml', 'r'))
    config.model.update(model_c.model)


def get_save_dir(config):
  return _SAVE_DIR.value + '_' + config.model.save_dir_name


def main(_):
  config = common_configs.get_config()
  config.model.update(_MODEL_CONFIG.value)
  config.sampler.update(_SAMPLER_CONFIG.value)

  update_sampler_cfg(config)
  sampler_mod = importlib.import_module(
      'discs.samplers.%s' % config.sampler.name
  )
  sampler = sampler_mod.build_sampler(config)
  logging.info(config)

  update_model_cfg(config)
  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  model = model_mod.build_model(config)

  experiment = experiment_mod.build_experiment(config)
  evaluator_mod = importlib.import_module(
      'discs.evaluators.%s' % config.experiment.evaluator
  )
  evaluator = evaluator_mod.build_evaluator(config)
  saver = saver_mod.build_saver(get_save_dir(config), config)

  metrics, running_time, acc_ratio, hops = experiment.get_results(
      model, sampler, evaluator
  )
  saver.save_results(acc_ratio, hops, metrics, running_time)


if __name__ == '__main__':
  app.run(main)
