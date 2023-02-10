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
import pdb

_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')

FLAGS = flags.FLAGS
_SAVE_DIR = flags.DEFINE_string('save_dir', './discs/results', 'Saving Dir')
_WEIGHT_FN = flags.DEFINE_string('weight_fn','SQRT', 'Balancing FN TYPE') 

def main(_):
  config = common_configs.get_config()
  config.model.update(_MODEL_CONFIG.value)
  config.sampler.update(_SAMPLER_CONFIG.value)

  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  logging.info(config)
  if config.model.name == 'rbm':
    model_path = f'{config.model.model_dir}/{config.model.dataset}-{config.model.num_categories}-{config.model.num_hidden}/rbm.pkl'
    model = pickle.load(open(model_path, 'rb'))
    config.model.num_visible = model['num_visible']
    config.model.data_mean = model['data_mean']
    config.model.shape = (model['num_visible'],)
    if model['params']['w'].shape[0] != model['num_visible']:
        model['params']['w'] = jnp.transpose(model['params']['w'])
    config.model.params = model['params']

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
  chain, num_loglike_calls, _ = experiment.get_batch_of_chains(model, sampler)
  running_time = time.time() - start_time

  chain = chain[
      int(config.experiment.chain_length * config.experiment.ess_ratio) :
  ]

  ess_metrcis = evaluator.get_effective_sample_size_metrics(
      chain, running_time, num_loglike_calls
  )

  if config.model.name == 'potts':
      dir_name = f'potts_{config.model.num_categories}'
  elif config.model.name == 'ising':
      if config.model.mu == 0.5:
        dir_name = 'ising_hightemp'
      elif config.model.mu == 1:
          dir_name = 'ising_lowtemp'
      else:
          dir_name = 'ising'
  elif config.model.name == 'categorical':
      dir_name = f'categorical_{config.model.num_categories}'
  elif self.config.model.name == 'rbm':
      if self.config.model.num_categories == 2:
          if self.config.model.num_hidden == 200:
              dir_name = 'rbm_lowtemp'
          else:
              dir_name = 'rbm_hightemp'
      else:
          dir_name = f'rbm_{self.config.model.num_categories}' 
 else:
      dir_name = config.model.name

  evaluator.save_results(_SAVE_DIR.value+'_'+dir_name, ess_metrcis, running_time)


if __name__ == '__main__':
  app.run(main)
