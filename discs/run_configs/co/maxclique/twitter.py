"""Config for maxclique twitter."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='maxclique',
          sampler='path_auxiliary',
          graph_type='twitter',
          sweep=[
              {
                  'config.experiment.chain_length': [1000, 800, 500],
              },
          ],
      )
  )
  return config
