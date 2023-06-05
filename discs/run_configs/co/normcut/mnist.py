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
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'sampler_config.name': [
                      'dmala',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.adaptive': [True, False],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'sampler_config.name': [
                      'path_auxiliary',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.adaptive': [True],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'sampler_config.name': [
                      'path_auxiliary',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.adaptive': [False],
                  'sampler_config.num_flips': [2],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'sampler_config.name': [
                      'gwg',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.solver': ['interpolate'],
                  'sampler_config.adaptive': [False],
                  'sampler_config.n': [500],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.solver': ['euler_forward'],
                  'sampler_config.adaptive': [False],
                  'sampler_config.n': [10000],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.chain_length': [500000],
                  'config.experiment.decay_rate': [0.05],
                  'config.experiment.init_temperature': [5],
                  'model_config.cfg_str': [
                      'r-MNIST',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
                  'sampler_config.adaptive': [True],
              },
          ],
      )
  )
  return config
