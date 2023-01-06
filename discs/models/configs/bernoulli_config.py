"""Config file for bernoulli models."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(100,),
      num_categories=2,
      init_sigma=0.5,
      name='bernoulli',
  )
  return config_dict.ConfigDict(model_config)
