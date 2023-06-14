from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='mis',
          graph_type='satlib',
          sampler='path_auxiliary',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.final_temperature': [0.00001],
                  'config.experiment.init_temperature': [1],
                  'config.experiment.chain_length': [500000],
              },
              {
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.final_temperature': [0.00001],
                  'config.experiment.init_temperature': [1],
                  'config.experiment.chain_length': [500000],
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
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.final_temperature': [0.00001],
                  'config.experiment.init_temperature': [1],
                  'config.experiment.chain_length': [500000],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.final_temperature': [0.00001],
                  'config.experiment.init_temperature': [1],
                  'config.experiment.chain_length': [500000],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
