from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='bernoulli',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [1000000],
                  'model_config.shape': [
                      '(2000,)',
                      '(10000,)',
                      '(100000,)',
                      '(500000,)',
                  ],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'config.experiment.chain_length': [1000000],
                  'model_config.shape': [
                      '(2000,)',
                      '(10000,)',
                      '(100000,)',
                      '(500000,)',
                  ],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'config.experiment.chain_length': [1000000],
                  'model_config.shape': [
                      '(2000,)',
                      '(10000,)',
                      '(100000,)',
                      '(500000,)',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
          ],
      )
  )
  return config
