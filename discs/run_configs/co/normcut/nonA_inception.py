# TODO:  'r-TRANSFORMER' is not working
from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='normcut',
          graph_type='nets',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [800000],
                  'config.experiment.init_temperature': [2],
                  'config.experiment.decay_rate': [0.15],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [800000],
                  'config.experiment.init_temperature': [2],
                  'config.experiment.decay_rate': [0.15],
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                  ],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.adaptive': [False],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [800000],
                  'config.experiment.init_temperature': [2],
                  'config.experiment.decay_rate': [0.15],
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'sampler_config.adaptive': [False],
              },
          ],
      )
  )
  return config
