"""Config for ba job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='maxcut',
          sampler='path_auxiliary',
          graph_type='ba',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                      'dlmc',
                  ],
                  'model_config.cfg_str': ['r-ba-4-n-1024-1100'],
              },
          ],
      )
  )
  return config
