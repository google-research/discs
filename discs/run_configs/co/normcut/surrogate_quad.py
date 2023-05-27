"""Config for normcut job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(dict(
      model='normcut',
      graph_type='nets',
      sampler='path_auxiliary',
      sweep=[
          {
              'model_config.cfg_str': 'r-INCEPTION',
              'model_config.stype': ['quad'],
              'config.experiment.decay_rate': [0.1, 0.15, 0.2],
              'config.experiment.t_schedule': ['exp_decay'],
              'config.experiment.batch_size': [32],
              'config.experiment.chain_length': [800000],
              'model_config.penalty': [0.0005, 0.001],
              'config.experiment.log_every_steps': [100],
              'config.experiment.init_temperature': [2, 5]},
          {
              'model_config.cfg_str': 'r-RESNET',
              'model_config.stype': ['quad'],
              'config.experiment.decay_rate': [0.1, 0.15, 0.2],
              'config.experiment.t_schedule': ['exp_decay'],
              'config.experiment.batch_size': [32],
              'config.experiment.chain_length': [800000],
              'model_config.penalty': [0.0005, 0.001],
              'config.experiment.log_every_steps': [100],
              'config.experiment.init_temperature': [2, 5]}
      ]
  ))
  return config