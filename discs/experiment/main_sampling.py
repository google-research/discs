"""Main script for sampling based experiments."""
import importlib
from absl import app
from absl import flags
from discs.common import configs as common_configs
from discs.common import utils
import discs.common.experiment_saver as saver_mod
from ml_collections import config_flags


FLAGS = flags.FLAGS
_EXPERIMENT_CONFIG = config_flags.DEFINE_config_file(
    'config', './discs/common/configs.py'
)
_MODEL_CONFIG = config_flags.DEFINE_config_file('model_config')
_SAMPLER_CONFIG = config_flags.DEFINE_config_file('sampler_config')
_RUN_LOCAL = flags.DEFINE_boolean('run_local', False, 'if runnng local')


def update_save_dir(config):
  if _RUN_LOCAL.value:
    save_folder = config.model.get('save_dir_name', config.model.name)
    save_root = './discs/results/' + save_folder
    config.experiment.save_root = save_root


def get_main_config():
  """Merge experiment, model and sampler config."""
  config = common_configs.get_config()
  if (
      'graph_type' not in _MODEL_CONFIG.value
      and 'bert_model' not in _MODEL_CONFIG.value
  ):
    config.update(_EXPERIMENT_CONFIG.value)
  config.sampler.update(_SAMPLER_CONFIG.value)
  config.model.update(_MODEL_CONFIG.value)
  if config.model.get('graph_type', None):
    graph_config = importlib.import_module(
        'discs.models.configs.%s.%s'
        % (config.model['name'], config.model['graph_type'])
    )
    config.model.update(graph_config.get_model_config(config.model['cfg_str']))
    co_exp_default_config = importlib.import_module(
        'discs.experiment.configs.co_experiment'
    )
    config.experiment.update(co_exp_default_config.get_co_default_config())
    config.update(_EXPERIMENT_CONFIG.value)

  if config.model.get('bert_model', None):
    config.update(_EXPERIMENT_CONFIG.value)

  return config


def main(_):
  config = get_main_config()
  update_save_dir(config)
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
      importlib.import_module('discs.experiment.sampling'),
      f'{config.experiment.name}',
  )
  experiment = experiment_mod(config)

  # evaluator
  evaluator_mod = importlib.import_module(
      'discs.evaluators.%s' % config.experiment.evaluator
  )
  evaluator = evaluator_mod.build_evaluator(config)

  # saver
  saver = saver_mod.build_saver(config)

  # chain generation
  experiment.get_results(model, sampler, evaluator, saver)


if __name__ == '__main__':
  app.run(main)
