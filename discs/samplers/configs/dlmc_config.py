"""Config for DLMC sampler."""

from discs.samplers.locallybalanced import LBWeightFn
from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='dlmc',
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type=LBWeightFn.SQRT,
      schedule_step=200,
      reset_z_est=20,
      solver='interpolate',
  )
  return config_dict.ConfigDict(model_config)
