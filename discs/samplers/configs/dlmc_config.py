"""Config for GWG sampler."""

from discs.samplers.locallybalanced import LBWeightFn
from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='dlmc',
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type=LBWeightFn.SQRT,
      logz_ema=0,
      reset_z_est=20,
  )
  return config_dict.ConfigDict(model_config)
