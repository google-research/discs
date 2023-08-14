"""Experiment config for maxcut er dataset."""
from ml_collections import config_dict


def get_config():
  """Get config."""
  exp_config = dict(
      experiment=dict(
          batch_size=16,
          t_schedule='linear',
          chain_length=10000,
          log_every_steps=100,
          init_temperature=1.0,
          decay_rate=0.1,
          final_temperature= 0.5,
          save_root='',
      )
  )
  return config_dict.ConfigDict(exp_config)