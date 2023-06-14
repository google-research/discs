"""Config file for resnet."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
          name='resnet',
          visualize = True,
          data_path='./BINARY_EBM/dynamic_mnist-64/',
          num_categories=2,
          )
  return config_dict.ConfigDict(model_config)

