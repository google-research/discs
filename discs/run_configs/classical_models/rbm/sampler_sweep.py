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
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [2],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-200/',
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-25/',
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [2],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-200/',
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-25/',
                  ],
                  'sampler_config.name': ['path_auxiliary', 'gwg', 'dmala'],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [2],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-200/',
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/mnist-2-25/',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [4],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/fashion_mnist-4-50/',
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [4],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/fashion_mnist-4-50/',
                  ],
                  'sampler_config.name': ['path_auxiliary', 'gwg', 'dmala'],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [4],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/fashion_mnist-4-50/',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [8],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/fashion_mnist-8-50/',
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [8],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/fashion_mnist-8-50/',
                  ],
                  'sampler_config.name': ['path_auxiliary', 'gwg', 'dmala'],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.save_samples': [True],
                  'model_config.num_categories': [8],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/RBM_DATA/fashion_mnist-8-50/',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
