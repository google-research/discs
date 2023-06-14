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
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.chain_length': [50000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.10',
                      'r-0.20',
                      'r-0.25',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'model_config.penalty': [1.001, 1.01, 1.1, 2],
              },
              {
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.chain_length': [50000],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.10',
                      'r-0.20',
                      'r-0.25',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'model_config.penalty': [1.001, 1.01, 1.1, 2],
              },
              {
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.chain_length': [50000],
                  'sampler_config.name': [
                      'gwg',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.10',
                      'r-0.20',
                      'r-0.25',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.adaptive': [False],
                  'model_config.penalty': [1.001, 1.01, 1.1, 2],
              },
              {
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.chain_length': [50000],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.cfg_str': [
                      'r-0.05',
                      'r-0.10',
                      'r-0.20',
                      'r-0.25',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'model_config.penalty': [1.001, 1.01, 1.1, 2],
              },
          ],
      )
  )
  return config
