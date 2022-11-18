"""Main script for sampling based experiments."""

from absl import app
from absl import flags
from absl import logging

import importlib
from ml_collections import config_flags
from discs.common import configs as common_configs

_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')


def main(_):
  config = common_configs.get_config()
  config.model.update(_MODEL_CONFIG.value)
  config.sampler.update(_SAMPLER_CONFIG.value)
  logging.info(config)

  model_mod = importlib.import_module('discs.models.%s' % config.model.name)
  model = model_mod.build_model(config)
  sampler_mod = importlib.import_module('discs.samplers.%s' % config.sampler.name)
  sampler = sampler_mod.build_sampler(config)


if __name__ == '__main__':
  app.run(main)
