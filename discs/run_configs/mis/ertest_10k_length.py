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
                  'config.experiment.chain_length': [
                      25000,
                      50000,
                      100000,
                      200000,
                      300000,
                      400000,
                  ],
                  'model_config.cfg_str': 'r-10k',
                  'config.experiment.final_temperature': [0.0001],
                  'config.experiment.t_schedule': ['linear'],
              },
              {
                  'config.experiment.chain_length': [
                      25000,
                      50000,
                      100000,
                      200000,
                      300000,
                      400000,
                  ],
                  'model_config.cfg_str': 'r-10k',
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.t_schedule': ['exp_decay'],
              },
          ],
      )
  )
  return config
