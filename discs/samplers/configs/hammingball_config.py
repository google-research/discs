"""Config for HammingBall sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='hammingball',
      block_size=10,
  )
  return config_dict.ConfigDict(sampler_config)
