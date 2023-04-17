"""Config for maxcut ba dataset."""

from ml_collections import config_dict


def get_config():
  """Get config."""
  exp_config = dict(
      experiment=dict(
          num_models=200,
          batch_size=16,
          t_schedule='exp_decay',
          chain_length=1001,
          log_every_steps=100,
          init_temperature=1.0,
          decay_rate=0.05,
          final_temperature=0.01,
      )
  )
  return config_dict.ConfigDict(exp_config)
