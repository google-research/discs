"""Config for Gibbs sampler."""

from discs.samplers.locallybalanced import LBWeightFn
from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='gibbs',
  )
  return config_dict.ConfigDict(model_config)
