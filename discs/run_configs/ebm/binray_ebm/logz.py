"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='resnet',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.name':['EBM_Experiment'],
                  'config.experiment.chain_length': [10000],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/caltech-64/',
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/dynamic_mnist-64/',
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/omniglot-64/',
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'config.experiment.name':['EBM_Experiment'],
                  'config.experiment.chain_length': [10000],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/caltech-64/',
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/dynamic_mnist-64/',
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/omniglot-64/',
                  ],
                  'sampler_config.name': ['path_auxiliary', 'gwg', 'dmala'],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
              {
                  'config.experiment.name':['EBM_Experiment'],
                  'config.experiment.chain_length': [10000],
                  'model_config.data_path': [
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/caltech-64/',
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/dynamic_mnist-64/',
                      '/gcs/xcloud-shared/kgoshvadi/data/BINARY_EBM/omniglot-64/',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
