"""Config for ertest job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='mis',
          sampler='path_auxiliary',
          sweep=[{
              'config.experiment.num_models': [128],
              'config.experiment.chain_length': [50000],
              'config.experiment.decay_rate': [0.01],
              'config.experiment.t_schedule': ['exp_decay'],
          }],
      )
  )
  return config
