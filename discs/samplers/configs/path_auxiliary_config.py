"""Config for Path Auxiliary sampler."""

from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='path_auxiliary',
      use_fast_path=True,
      num_flips=1,
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type='SQRT',
      approx_with_grad=True,
  )
  return config_dict.ConfigDict(sampler_config)
