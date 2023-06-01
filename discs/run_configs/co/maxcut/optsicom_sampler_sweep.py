from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='maxcut',
          sampler='path_auxiliary',
          graph_type='optsicom',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': [
                      'r-b',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'config.experiment.num_models': [10],
                  'config.experiment.batch_size': [16],
              },
              {
                  'config.experiment.num_models': [10],
                  'config.experiment.batch_size': [16], 
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                  ],
                  'model_config.cfg_str': [
                      'r-b',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'config.experiment.num_models': [10],
                  'config.experiment.batch_size': [16], 
                  'sampler_config.name': [
                      'gwg',
                  ],
                  'model_config.cfg_str': [
                      'r-b',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.adaptive': [False],
              },
              {
                  'config.experiment.num_models': [10],
                  'config.experiment.batch_size': [16], 
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.cfg_str': [
                      'r-b',
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
