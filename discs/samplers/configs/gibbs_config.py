"""Config for Gibbs sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='gibbs',
  )
  return config_dict.ConfigDict(sampler_config)
