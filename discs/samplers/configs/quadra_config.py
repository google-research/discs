"""Config for DLMC sampler."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='quadra',
      diag_type='shift',
  )
  return config_dict.ConfigDict(model_config)
