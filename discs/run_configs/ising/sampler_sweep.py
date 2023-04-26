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
                  'sampler_config.name': [
                      'randomwalk',
                  ],
                  'model_config.mu': [0.5, 1],
                  'model_config.lambdaa': [0.5, 1],
                  'model_config.sigma': [1.5, 3],
              },
              {
                  'sampler_config.name': [
                      'path_auxiliary',
                      'dlmc',
                      'gwg',
                  ],
                  'model_config.mu': [0.5, 1],
                  'model_config.lambdaa': [0.5, 1],
                  'model_config.sigma': [1.5, 3],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'], 
              },
          ],
      )
  )
  return config
