"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='potts',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.lambdaa': [1],
                  'model_config.init_sigma': [1.5],
                  'model_config.mu': [0.5],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'model_config.lambdaa': [1],
                  'model_config.init_sigma': [1.5],
                  'model_config.mu': [0.5],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.lambdaa': [1],
                  'model_config.init_sigma': [1.5],
                  'model_config.mu': [0.5],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.lambdaa': [2],
                  'model_config.init_sigma': [3],
                  'model_config.mu': [1],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'model_config.lambdaa': [2],
                  'model_config.init_sigma': [3],
                  'model_config.mu': [1],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.lambdaa': [2],
                  'model_config.init_sigma': [3],
                  'model_config.mu': [1],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.lambdaa': [3],
                  'model_config.init_sigma': [4.5],
                  'model_config.mu': [1.5],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'model_config.lambdaa': [3],
                  'model_config.init_sigma': [4.5],
                  'model_config.mu': [1.5],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.lambdaa': [3],
                  'model_config.init_sigma': [4.5],
                  'model_config.mu': [1.5],
              },
          ],
      )
  )
  return config
