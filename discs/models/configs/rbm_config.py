"""Config file for rbms."""

from ml_collections import config_dict


def get_config():
  c = dict(
          name='rbm',
          data_path='./discs/storage/models/rbm/mnist-2-200/'
          )
  return config_dict.ConfigDict(c)
