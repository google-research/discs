"""Config for BlockGibbs sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict (
      block_size=2,
      name='blockgibbs',
  )
  return config_dict.ConfigDict(sampler_config)
