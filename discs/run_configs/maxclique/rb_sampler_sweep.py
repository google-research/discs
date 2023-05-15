"""Config for rb job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='maxclique',
          sampler='path_auxiliary',
          graph_type='rb',
          sweep=[
              {
                  'config.experiment.decay_rate': [0.1, 0.05, 0.01],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.batch_size': [16],
                  'config.experiment.chain_length': [
                      1001,
                      2001,
                      3001,
                      4001,
                      5001,
                  ],
                  'config.experiment.init_temperature': [1.0, 0.5],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
              },
              {
                  'config.experiment.decay_rate': [0.1, 0.05, 0.01],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.batch_size': [16],
                  'config.experiment.chain_length': [
                      1001,
                      2001,
                      3001,
                      4001,
                      5001,
                  ],
                  'config.experiment.init_temperature': [1.0, 0.5],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
              },
              {
                  'config.experiment.decay_rate': [0.1, 0.05, 0.01],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.batch_size': [16],
                  'config.experiment.chain_length': [
                      1001,
                      2001,
                      3001,
                      4001,
                      5001,
                  ],
                  'config.experiment.init_temperature': [1.0, 0.5],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'sampler_config.balancing_fn_type': ['SQRT', 'RATIO'],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
