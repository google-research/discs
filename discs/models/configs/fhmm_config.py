"""Config file for FHMM."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(200, 50),  # (L, K)
      num_categories=2,
      sigma=2,
      alpha=0.1,
      beta=0.8,
      name='fhmm',
  )
  model_config['save_dir_name'] = 'fhmm'
  return config_dict.ConfigDict(model_config)
