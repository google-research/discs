"""Config for categorical num categories sweep."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='categorical',
          sampler='dlmc',
          sweep=[
              {
                  'model_config.num_categories': [4, 8, 16, 32],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'model_config.init_sigma': [1.5],
                  'sampler_config.name': ['dlmc'],
              },
          ],
      )
  )
  return config
