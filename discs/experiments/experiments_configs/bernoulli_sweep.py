"""Config for model sweeps job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='bernoulli',
          sampler='randomwalk',
          sweep=[
              {'config.model.sigma': [0.5, 1.5]},
          ],
      )
  )
  return config
