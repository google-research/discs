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
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.num_categories': [4, 8, 18, 32, 64],
              },
              {
                  'config.experiment.chain_length': [100000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.num_categories': [4, 8, 18, 32, 64],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'config.experiment.chain_length': [100000],,
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.num_categories': [4, 8, 18, 32, 64],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
          ],
      )
  )
  return config
