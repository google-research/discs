"""Config for maxcut ba dataset."""

from discs.common import configs
from ml_collections import config_dict
from sco.experiments import default_configs


def get_config(cfg_str):
  """Get config."""
  exp_config = dict(
    samples_per_instance = 16,
    t_schedule = 'linear',
    chain_length = 10000,
    log_every_steps = 1,
    init_temperature = 1,
    decay_rate = 0.1,
    final_temperature = 0.5,
    approx_with_grad = False
  )
  config.experiment.update(default_configs.get_exp_config())

  config.experiment.batch_size = 1024
  if num_nodes >= 300:
    config.experiment.batch_size = 128


  return config_dict.ConfigDict(exp_config)