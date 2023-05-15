"""Config for bernoulli dimension sweep."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='bernoulli',
          sampler='dlmc',
          sweep=[
              {
                  'model_config.shape': [
                      '(1000,)',
                      '(10000,)',
                      '(50000,)',
                      '(100000,)',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
          ],
      )
  )
  return config
