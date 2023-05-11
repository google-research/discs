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
                  'config.experiment.chain_length': [
                      5000,
                      10000,
                      20000,
                      30000,
                      40000,
                      50000,
                  ],
                  'model_config.cfg_str': ['r-ba-4-n-1024-1100'],
              },
          ],
      )
  )
  return config
