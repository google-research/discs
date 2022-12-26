"""Config file for rbms."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
      num_visible=0,
      num_hidden=200,
      num_categories=2,
      name='rbm',
  )
  return config_dict.ConfigDict(model_config)
