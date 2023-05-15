"""Config for ba job."""

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
                  'config.experiment.chain_length': [200000],
                  'model_config.cfg_str': [
                      'r-ba-4-n-1024-1100',
                      'r-ba-4-n-512-600',
                      'r-ba-4-n-256-300',
                  ],
                  'config.experiment.decay_rate': [0.01, 0.1],
                  'config.experiment.init_temperature': [0.1, 0.5, 1],
                  'config.experiment.log_every_steps': [100],
                  
              },
          ],
      )
  )
  return config
