"""Config for maxcut ba dataset."""

from discs.common import configs
from ml_collections import config_dict
from sco.experiments import default_configs


def get_config(cfg_str):
  """Get config."""
  exp_config = dict(
      batch_size=1024,
      samples_per_instance=16,
      t_schedule='exp_decay',
      chain_length=10000,
      log_every_steps=10,
      init_temperature=0.5,
      decay_rate=0.05,
      final_temperature=0.000001,
  )
  return config_dict.ConfigDict(exp_config)
