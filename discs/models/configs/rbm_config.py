"""Config file for rbms."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
          name='rbm',
          visualize = True,
          data_path='./RBM_DATA/fashion_mnist-4-50/',
          num_categories=4,
          )
  return config_dict.ConfigDict(model_config)
