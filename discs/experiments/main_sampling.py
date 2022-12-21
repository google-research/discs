"""Main script for sampling based experiments."""
import importlib
from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags
from discs.common import configs as common_configs
from discs.experiment import experiment as experiment_mod
from discs.evaluation import evaluator as evaluator_mod
import time


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
  
  evaluator.save_results(_SAVE_DIR.value, ess_metrcis, running_time)


if __name__ == '__main__':
  app.run(main)
