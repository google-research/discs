from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='mis',
          sampler='path_auxiliary',
          graph_type='er_density',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.1',
                      'r-0.15',
                      'r-0.2',
                      'r-0.25',
                  ],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.1',
                      'r-0.15',
                      'r-0.2',
                      'r-0.25',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'sampler_config.name': [
                      'gwg',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.1',
                      'r-0.15',
                      'r-0.2',
                      'r-0.25',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.adaptive': [False],
              },
              {
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.1',
                      'r-0.15',
                      'r-0.2',
                      'r-0.25',
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
