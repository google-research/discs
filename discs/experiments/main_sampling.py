"""Main script for sampling based experiments."""
import importlib
from absl import app
from absl import flags
from absl import logging

from discs.samplers.locallybalanced import LBWeightFn
from ml_collections import config_flags
from discs.common import configs as common_configs
from discs.experiment import experiment as experiment_mod
from discs.evaluation import evaluator as evaluator_mod
import time
import os
import pickle
import jax.numpy as jnp
import yaml
import pdb

_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')

FLAGS = flags.FLAGS
_SAVE_DIR = flags.DEFINE_string('save_dir', './discs/results', 'Saving Dir')
_WEIGHT_FN = flags.DEFINE_string('weight_fn', 'SQRT', 'Balancing FN TYPE')


def main(_):
  config = common_configs.get_config()
  config.model.update(_MODEL_CONFIG.value)
  config.sampler.update(_SAMPLER_CONFIG.value)

  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  logging.info(config)
  if config.model.name == 'rbm':
    model = pickle.load(open(config.model.data_path + 'params.pkl', 'rb'))
    config.model.params = model['params']
    config.model.train = False
    model_c = yaml.unsafe_load(open(config.model.data_path+ 'config.yaml', 'r'))
    config.model.update(model_c.model)

  model = model_mod.build_model(config)
  sampler_mod = importlib.import_module(
      'discs.samplers.%s' % config.sampler.name
  )
  if 'balancing_fn_type' in config.sampler.keys():
    if _WEIGHT_FN.value == 'RATIO':
      config.sampler['balancing_fn_type'] = LBWeightFn.RATIO
    elif _WEIGHT_FN.value == 'MAX':
      config.sampler['balancing_fn_type'] = LBWeightFn.MAX
    elif _WEIGHT_FN.value == 'MIN':
      config.sampler['balancing_fn_type'] = LBWeightFn.MIN
    else:
      config.sampler['balancing_fn_type'] = LBWeightFn.SQRT

  sampler = sampler_mod.build_sampler(config)
  experiment = experiment_mod.build_experiment(config)
  evaluator = evaluator_mod.build_evaluator(config)

  start_time = time.time()
  chain, num_loglike_calls, acc_ratio, hops, _ = experiment.get_batch_of_chains(
      model, sampler
  )
  running_time = time.time() - start_time

  chain = chain[
      int(config.experiment.chain_length * config.experiment.ess_ratio) :
  ]

  ess_metrcis = evaluator.get_effective_sample_size_metrics(
      chain, running_time, num_loglike_calls
  )

  save_path = _SAVE_DIR.value + '_' + config.model.save_dir_name
  evaluator.save_results(save_path, ess_metrcis, running_time)
  evaluator.plot_acc_ratio(save_path, acc_ratio)
  evaluator.plot_hops(save_path, hops)


if __name__ == '__main__':
  app.run(main)
