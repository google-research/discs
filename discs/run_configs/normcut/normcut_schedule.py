"""Config for normcut job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='normcut',
          graph_type='nets',
          sweep=[
              {
                  'cfg_str': 'r-VGG',
                  'config.experiment.decay_rate': [0.1, 0.05, 0.01, 0.005],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.model.penalty': [0.0],
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [800000],
                  'config.sampler.approx_with_grad': [False, True],
                  'config.experiment.init_temperature': [0.002, 0.005, 0.01],
              },
              {
                  'cfg_str': 'r-MNIST',
                  'config.experiment.decay_rate': [0.1, 0.05, 0.01, 0.005],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.model.penalty': [0.0],
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [800000],
                  'config.sampler.approx_with_grad': [False, True],
                  'config.experiment.init_temperature': [0.002, 0.005, 0.01],
              },
              {
                  'cfg_str': 'r-ALEXNET',
                  'config.experiment.decay_rate': [0.1, 0.05, 0.01, 0.005],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.model.penalty': [0.0],
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [800000],
                  'config.sampler.approx_with_grad': [False, True],
                  'config.experiment.init_temperature': [0.002, 0.005, 0.01],
              },
          ],
      )
  )
  return config
