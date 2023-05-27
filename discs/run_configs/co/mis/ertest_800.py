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
              'config.experiment.num_models': [128],
              'config.experiment.chain_length': [50000, 100000, 200000],
              'config.experiment.decay_rate': [0.01, 0.05, 0.08],
              'config.experiment.t_schedule': ['exp_decay'],
              'config.experiment.log_every_steps': [100],
          }],
      )
  )
  return config