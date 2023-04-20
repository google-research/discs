"""Config for ertest job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='normcut',
          graph_type='nets',
          sampler='path_auxiliary',
          sweep=[{
              'config.experiment.decay_rate': [0.1, 0.05],
              'config.experiment.t_schedule': ['exp_decay'],
              'config.experiment.batch_size': [16],
              'config.experiment.chain_length': [800000],
              'config.experiment.init_temperature': [2, 5],
              'model_config.tolerance': [0.045, 0.048, 0.05, 0.052],
              'cfg_str': [
                  's-path_auxiliary,r-INCEPTION',
                  's-path_auxiliary,r-VGG',
                  's-path_auxiliary,r-RESNET',
              ],
          }],
      )
  )
  return config
