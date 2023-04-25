"""Config for satlib job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(dict(
      model='mis',
      graph_type='satlib',
      sweep=[
          {'config.experiment.t_schedule': ['exp_decay'],
           'config.experiment.decay_rate': [0.05, 0.08, 0.1],
           'config.experiment.chain_length': [100001, 200001, 500001, 1000001]},
      ]
  ))
  return config