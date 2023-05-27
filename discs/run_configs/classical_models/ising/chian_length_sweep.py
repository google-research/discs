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
                  'config.experiment.chain_length': [
                      100,
                      1000,
                      10000,
                      100000,
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.mu': [0.5],
                  'model_config.lambdaa': [0.5],
                  'model_config.init_sigma': [1.5],
              },
              {
                  'config.experiment.chain_length': [
                      100,
                      1000,
                      10000,
                      100000,
                  ],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.mu': [0.5],
                  'model_config.lambdaa': [0.5],
                  'model_config.init_sigma': [1.5],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
              {
                  'config.experiment.chain_length': [
                      100,
                      1000,
                      10000,
                      100000,
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.mu': [0.5],
                  'model_config.lambdaa': [0.5],
                  'model_config.init_sigma': [1.5],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
