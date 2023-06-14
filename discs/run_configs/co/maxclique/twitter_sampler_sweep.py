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
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'sampler_config.adaptive': [False],
                  'sampler_config.name': [
                      'gwg',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
