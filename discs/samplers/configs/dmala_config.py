"""Config for DMALA sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='dmala',
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type='SQRT',
      schedule_step=100,
      reset_z_est=20,
      step_size=0.2,
  )
  return config_dict.ConfigDict(sampler_config)
