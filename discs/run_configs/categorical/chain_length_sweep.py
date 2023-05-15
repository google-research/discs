from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='categorical',
          sampler='dlmc',
          sweep=[
              {
                  'config.experiment.chain_length': [10000, 100000, 500000],
                  'sampler_config.balancing_fn_type': ['SQRT'],
                  'model_config.init_sigma': [1.5],
                  'model_config.num_categories': [4, 8],
                  'sampler_config.name': ['dlmc'],
              },
          ],
      )
  )
  return config
