"""Main script for sampling based experiments."""
import importlib
from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
from discs.common import configs as common_configs
from discs.experiment import experiment as experiment_mod
from discs.evaluation import evaluator as evaluator_mod
import numpy as np
import time
import os
import pdb
import csv

_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')

FLAGS = flags.FLAGS
_SAVE_DIR = flags.DEFINE_string('save_dir', './discs/results', 'Saving Dir')


def main(_):
  config = common_configs.get_config()
  config.model.update(_MODEL_CONFIG.value)
  config.sampler.update(_SAMPLER_CONFIG.value)
  logging.info(config)

  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  model = model_mod.build_model(config)
  sampler_mod = importlib.import_module(
      'discs.samplers.%s' % config.sampler.name
  )
  sampler = sampler_mod.build_sampler(config)
  experiment = experiment_mod.build_experiment(config)
  evaluator = evaluator_mod.build_evaluator(config)

  start_time = time.time()
  chain, num_loglike_calls, _ = experiment.get_batch_of_chains(model, sampler)
  running_time = time.time() - start_time

  chain = chain[
      int(config.experiment.chain_length * config.experiment.ess_ratio) :
  ]

  ess_metrcis = evaluator.get_effective_sample_size_metrics(
      chain, running_time, num_loglike_calls
  )

  if not os.path.isdir(_SAVE_DIR.value):
    os.makedirs(_SAVE_DIR.value)

  results = {}

  results['sampler'] = config.sampler.name
  if 'adaptive' in config.sampler.keys():
    results['sampler'] = f'a_{config.sampler.name}'

  if 'balancing_fn_type' in config.sampler.keys():
    if config.sampler.balancing_fn_type == 2:
      results['sampler'] = results['sampler'] + '(frac)'
    elif config.sampler.balancing_fn_type == 3:
      results['sampler'] = results['sampler'] + '(and)'
    elif config.sampler.balancing_fn_type == 4:
      results['sampler'] = results['sampler'] + '(or)'
    else:
      results['sampler'] = results['sampler'] + '(sqrt)'

  ess_metrcis = np.array(ess_metrcis)
  results['model'] = config.model.name
  results['num_categories'] = config.model.num_categories
  results['shape'] = config.model.shape
  results['ESS'] = ess_metrcis[0]
  results['ESS_M-H'] = ess_metrcis[1]
  results['ESS_T'] = ess_metrcis[2]
  results['ESS_EE'] = ess_metrcis[3]
  results['Time'] = running_time
  results['batch_size'] = config.experiment.batch_size
  results['chain_length'] = config.experiment.chain_length
  results['ess_ratio'] = config.experiment.ess_ratio


  csv_path = f'{_SAVE_DIR.value}/results.csv'
  if not os.path.exists(csv_path):
    with open(f'{_SAVE_DIR.value}/results.csv', 'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
      writer.writeheader()
      writer.writerow(results)
      csvfile.close()
  else:
    with open(f'{_SAVE_DIR.value}/results.csv', 'a') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=list(results.keys()))
      writer.writerow(results)
      csvfile.close()

  with open(
      f'{_SAVE_DIR.value}/{config.model.name}_{config.sampler.name}_{running_time}.txt',
      'w',
  ) as f:
    f.write('Mean ESS: {} \n'.format(ess_metrcis[0]))
    f.write('ESS M-H Steps: {} \n'.format(ess_metrcis[1]))
    f.write('ESS over time: {} \n'.format(ess_metrcis[2]))
    f.write('ESS over loglike calls: {} \n'.format(ess_metrcis[3]))
    f.write('Running time: {} s \n'.format(running_time))
    f.write(str(config))


if __name__ == '__main__':
  app.run(main)
