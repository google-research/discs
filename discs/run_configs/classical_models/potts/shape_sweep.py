"""Config for categorical dimension sweep."""

from ml_collections import config_dict




def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='potts',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [1000000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.shape': [
                      '(10, 10)',
                      '(30, 30)',
                      '(100, 100)',
                      '(300, 300)',
                  ],
              },
              {
                  'config.experiment.chain_length': [1000000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'model_config.shape': [
                      '(10, 10)',
                      '(30, 30)',
                      '(100, 100)',
                      '(300, 300)',
                  ],
              },
              {
                  'config.experiment.chain_length': [1000000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.shape': [
                      '(10, 10)',
                      '(30, 30)',
                      '(100, 100)',
                      '(300, 300)',
                  ],
              },
          ],
      )
  )
  return config
