"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='categorical',
          sampler='path_auxiliary',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'hammingball'
                  ],
                  'model_config.init_sigma': [1.5],
                  'model_config.num_categories': [4, 8],
              },
              {
                  'sampler_config.name': [
                      'path_auxiliary',
                  ],
                  'model_config.init_sigma': [1.5],
                  'model_config.num_categories': [4, 8],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
          ],
      )
  )
  return config
