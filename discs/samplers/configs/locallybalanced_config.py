from discs.samplers.locallybalanced import LBWeightFn
from ml_collections import config_dict


def get_config():
  model_config = dict(
      balancing_fn_type=LBWeightFn.SQRT,
      name='locallybalanced',
  )
  return config_dict.ConfigDict(model_config)
