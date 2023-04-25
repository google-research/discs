"""Config file for rbms."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
          name='rbm',
          visualize = True,
          data_path='gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/,
          num_categories=2,
          )
  return config_dict.ConfigDict(model_config)
