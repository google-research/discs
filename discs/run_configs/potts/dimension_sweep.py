"""Config for categorical dimension sweep."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='potts',
          sampler='dlmc',
          sweep=[
              {
                  'model_config.shape': [
                      '(10, 10)',
                      '(30, 30)',
                      '(100, 100)',
                      '(500, 500)',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
          ],
      )
  )
  return config
