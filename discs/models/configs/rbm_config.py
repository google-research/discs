"""Config file for rbms."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
          dataset='mnist',
          num_categories=2,
          num_hidden=25,
          num_visible=784,
          shape=(784,)
          name='rbm',
          train=False,
          save_dir_name='rbm_lowtemp',
          model_path='./discs/storage/models/rbm/mnist-2-200/rbm.pkl'
          )
  return config_dict.ConfigDict(model_config)
