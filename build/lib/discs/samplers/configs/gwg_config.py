"""Config for GWG sampler."""

from discs.samplers.locallybalanced import LBWeightFn
from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='gwg',
      num_flips=1,
      adaptive=True,
      target_acceptance_rate=0.574,
      balancing_fn_type=LBWeightFn.SQRT,
  )
  return config_dict.ConfigDict(model_config)
