from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='potts',
          sampler='dlmc',
          sweep=[
              {
                  'config.experiment.batch_size': [4, 8, 128, 512, 1024],
                  'sampler_config.balancing_fn_type': ['SQRT'],
              },
          ],
      )
  )
  return config
