"""Config file for bernoulli models."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(10000,),
      num_categories=2,
      init_sigma=0.5,
      name='bernoulli',
  )
  if model_config['init_sigma'] == 0.5:
    model_config['save_dir_name'] = 'bernoulli_hightemp'
  else:
    model_config['save_dir_name'] = 'bernoulli_lowtemp'

  return config_dict.ConfigDict(model_config)
