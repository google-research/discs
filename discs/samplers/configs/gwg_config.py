"""Config for GWG sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='gwg',
      num_flips=1,
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type='SQRT',
  )
  return config_dict.ConfigDict(sampler_config)
