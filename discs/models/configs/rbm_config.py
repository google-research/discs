"""Config file for rbms."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
          dataset='mnist',
          num_categories=2,
          num_hidden=200,
          name='rbm',
          train=False,
          model_dir = './discs/storage/models/rbm/'
          )
  return config_dict.ConfigDict(model_config)
