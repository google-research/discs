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
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.chain_length': [800000],
                  'config.experiment.save_every_steps': [100],
                  'config.experiment.init_temperature': [2],
                  'config.experiment.decay_rate': [0.15],
                  'config.experiment.final_temperature': [0.0000001],
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
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.chain_length': [800000],
                  'config.experiment.save_every_steps': [100],
                  'config.experiment.init_temperature': [2],
                  'config.experiment.decay_rate': [0.15],
                  'config.experiment.final_temperature': [0.0000001],
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
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
              {
                  'config.experiment.batch_size': [32],
                  'config.experiment.t_schedule': ['exp_decay'],
                  'config.experiment.chain_length': [800000],
                  'config.experiment.save_every_steps': [100],
                  'config.experiment.init_temperature': [2],
                  'config.experiment.decay_rate': [0.15],
                  'config.experiment.final_temperature': [0.0000001],
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                  ],
                  'sampler_config.name': [
                      'dlmc',
                  ],
                  'model_config.stype': ['quad'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
                  'sampler_config.solver': ['interpolate', 'euler_forward'],
              },
          ],
      )
  )
  return config
