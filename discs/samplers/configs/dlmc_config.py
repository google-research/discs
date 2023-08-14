"""Config for DLMC sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='dlmc',
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type='SQRT',
      schedule_step=200,
      reset_z_est=20,
      solver='interpolate',
      n=3.0,
  )
  return config_dict.ConfigDict(sampler_config)
