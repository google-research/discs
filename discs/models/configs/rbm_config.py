"""Config file for rbms."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
          dataset='mnist',
          num_categories=2,
          num_hidden=200,
          name='rbm',
          train=False,
          model_dir = './experiment/'
          #model_dir = './discs/storage/models/rbm/'
          )
  print("Loading the model from:", model_config['model_dir'])
  return config_dict.ConfigDict(model_config)
