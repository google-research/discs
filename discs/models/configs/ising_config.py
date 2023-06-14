"""Config file for ising model."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(50, 50),
      num_categories=2,
      lambdaa=0.5,
      external_field_type=1,
      mu=0.5,
      init_sigma=1.5,
      name='ising',
  )
  if model_config['lambdaa'] == 0.5:
    model_config['save_dir_name'] = 'ising_hightemp'
  else:
    model_config['save_dir_name'] = 'ising_lowtemp'

  return config_dict.ConfigDict(model_config)
