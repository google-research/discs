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
                  ],
                  'model_config.cfg_str': [
                      'r-ba-4-n-1024-1100',
                      'r-ba-4-n-512-600',
                      'r-ba-4-n-256-300',
                      'r-ba-4-n-128-150',
                      'r-ba-4-n-64-75',
                      'r-ba-4-n-32-40',
                      'r-ba-4-n-16-20',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'config.experiment.chain_length': [100000],
              },
              {
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                  ],
                  'model_config.cfg_str': [
                      'r-ba-4-n-1024-1100',
                      'r-ba-4-n-512-600',
                      'r-ba-4-n-256-300',
                      'r-ba-4-n-128-150',
                      'r-ba-4-n-64-75',
                      'r-ba-4-n-32-40',
                      'r-ba-4-n-16-20',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'config.experiment.chain_length': [100000],
              },
              {
                  'sampler_config.name': [
                      'gwg',
                  ],
                  'model_config.cfg_str': [
                      'r-ba-4-n-1024-1100',
                      'r-ba-4-n-512-600',
                      'r-ba-4-n-256-300',
                      'r-ba-4-n-128-150',
                      'r-ba-4-n-64-75',
                      'r-ba-4-n-32-40',
                      'r-ba-4-n-16-20',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.adaptive': [False],
                  'config.experiment.chain_length': [100000],
              },
              {
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.cfg_str': [
                      'r-ba-4-n-1024-1100',
                      'r-ba-4-n-512-600',
                      'r-ba-4-n-256-300',
                      'r-ba-4-n-128-150',
                      'r-ba-4-n-64-75',
                      'r-ba-4-n-32-40',
                      'r-ba-4-n-16-20',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'config.experiment.chain_length': [100000],
              },
          ],
      )
  )
  return config
