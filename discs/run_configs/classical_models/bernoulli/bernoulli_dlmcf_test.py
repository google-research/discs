from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='bernoulli',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [100000, 1000000],
                  'model_config.shape': [
                      '(1000000,)',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.solver': ['euler_forward'],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
          ],
      )
  )
  return config
