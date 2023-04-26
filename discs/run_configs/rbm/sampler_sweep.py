"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='rbm',
          sampler='path_auxiliary',
          sweep=[
              {
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-200/',
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-25/',
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                  ],
              },
              {
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-200/',
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-25/',
                  ],
                  'sampler_config.name': [
                      'path_auxiliary',
                      'dlmc',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
          ],
      )
  )
  return config
