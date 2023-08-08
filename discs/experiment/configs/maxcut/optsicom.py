"""Experiment config for maxcut optsicom dataset."""
from ml_collections import config_dict


def get_config():
  """Get config."""
  exp_config = dict(
      experiment=dict(
          batch_size=32,
          t_schedule='exp_decay',
          chain_length=25000,
          log_every_steps=100,
          init_temperature=1.0,
          decay_rate=0.05,
          final_temperature=0.001,
          save_root='',
      )
  )
  return config_dict.ConfigDict(exp_config)