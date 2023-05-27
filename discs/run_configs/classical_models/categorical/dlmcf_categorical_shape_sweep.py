from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='categorical',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [100000],
                  'model_config.shape': [
                      '(250,)',
                      '(2000,)',
                      '(32000,)',
                      '(512000,)',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.init_sigma': [1.5],
                  'model_config.num_categories': [4],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
          ],
      )
  )
  return config
