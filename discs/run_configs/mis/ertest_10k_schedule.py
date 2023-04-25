"""Config for ertest job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(dict(
      model='mis',
      graph_type='ertest',
      sweep=[
          {'config.experiment.chain_length': [400000],
           'cfg_str': 'r-10k',
           'config.experiment.final_temperature': [0.01, 0.001, 0.0001],
           'config.experiment.t_schedule': ['linear']},
          {'config.experiment.chain_length': [400000],
           'cfg_str': 'r-10k',
           'config.experiment.init_temperature': [0.01, 0.001, 0.0001],
           'config.experiment.t_schedule': ['constant']},
          {'config.experiment.chain_length': [400000],
           'cfg_str': 'r-10k',
           'config.experiment.init_temperature': [1.0, 0.1],
           'config.experiment.decay_rate': [0.05, 0.01, 0.005],
           'config.experiment.t_schedule': ['exp_decay']},
      ]
  ))
  return config