"""Config for ertest job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='mis',
          graph_type='ertest',
          sampler='path_auxiliary',
          sweep=[{
              'config.experiment.batch_size': [128],
              'config.experiment.chain_length': [50000, 100000, 200000],
              'config.experiment.decay_rate': [0.01, 0.05, 0.08],
              'config.experiment.t_schedule': ['exp_decay'],
          }],
      )
  )
  return config
