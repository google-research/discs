"""Config for ertest job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(dict(
      model='mis',
      graph_type='ertest',
      sweep=[
          {'config.experiment.bath_size': [
              1, 2, 4, 8, 16, 32, 64],
           'cfg_str': 'r-10k',
           'config.experiment.decay_rate': [0.05],
           'config.experiment.t_schedule': ['exp_decay']},
      ]
  ))
  return config