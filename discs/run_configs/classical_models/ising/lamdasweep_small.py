"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='ising',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [400000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0],
                  'model_config.init_sigma': [0],
                  'model_config.lambdaa': [
                      0.1607,
                      0.2007,
                      0.2407,
                      0.2807,
                      0.3207,
                      0.3607,
                      0.4007,
                      0.4407,
                      0.4807,
                      0.5207,
                      0.5607,
                      0.6007,
                      0.6407,
                      0.6807,
                      0.7207,
                      0.7607,
                  ],
              },
              {
                  'config.experiment.chain_length': [400000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0],
                  'model_config.init_sigma': [0],
                  'model_config.lambdaa': [
                      0.1607,
                      0.2007,
                      0.2407,
                      0.2807,
                      0.3207,
                      0.3607,
                      0.4007,
                      0.4407,
                      0.4807,
                      0.5207,
                      0.5607,
                      0.6007,
                      0.6407,
                      0.6807,
                      0.7207,
                      0.7607,
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.chain_length': [400000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0],
                  'model_config.init_sigma': [0],
                  'model_config.lambdaa': [
                      0.1607,
                      0.2007,
                      0.2407,
                      0.2807,
                      0.3207,
                      0.3607,
                      0.4007,
                      0.4407,
                      0.4807,
                      0.5207,
                      0.5607,
                      0.6007,
                      0.6407,
                      0.6807,
                      0.7207,
                      0.7607,
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
