"""Main script for sampling based experiments."""

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
  chain, samples, num_loglike_calls, _ = experiment.get_batch_of_chains(
      model, sampler
  )
  running_time = time.time() - start_time
  ess_metrcis = evaluator.get_effective_sample_size_metrics(
      samples, running_time, num_loglike_calls
  )


if __name__ == '__main__':
  app.run(main)
