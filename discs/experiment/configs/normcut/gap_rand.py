"""Experiment config for random graph generator."""

from ml_collections import config_dict


def get_config():
  """Get config for er benchmark graphs."""
  exp_config = dict(
      experiment=dict(
          batch_size=8,
          t_schedule='exp_decay',
          chain_length=500000,
          log_every_steps=100,
          save_every_steps=100,
          init_temperature=1,
          decay_rate=0.005,
          final_temperature=0.0000001,
          save_root='',
      )
  )

  return config_dict.ConfigDict(exp_config)