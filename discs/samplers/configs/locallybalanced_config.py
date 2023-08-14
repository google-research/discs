"""Config for LocallyBalanced sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      balancing_fn_type='SQRT',
      name='locallybalanced',
  )
  return config_dict.ConfigDict(sampler_config)
