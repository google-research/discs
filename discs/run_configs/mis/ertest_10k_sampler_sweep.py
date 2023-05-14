"""Config for ertest job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='mis',
          sampler='path_auxiliary',
          graph_type='ertest',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': ['r-10k'],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'model_config.cfg_str': ['r-10k'],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.cfg_str': ['r-10k'],
                  'config.experiment.log_every_steps': [100],
              },
          ],
      )
  )
  return config
