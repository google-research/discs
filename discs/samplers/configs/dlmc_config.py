"""Config for DLMC sampler."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='dlmc',
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type='SQRT',
      schedule_step=200,
      reset_z_est=20,
      solver='interpolate',
      step_size=0.1,
  )
  return config_dict.ConfigDict(model_config)
