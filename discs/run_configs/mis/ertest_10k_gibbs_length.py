"""Config for ertest job."""

from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(dict(
      model='mis',
      sampler='blockgibbs',
      graph_type='ertest',
      sweep=[
          {'config.experiment.chain_length': [
              1000, 2000, 3000, 4000],
           'model_config.cfg_str': 's-gibbs,r-10k',
           'config.experiment.final_temperature': [0.0001],
           'config.experiment.t_schedule': ['linear'],
           'config.experiment.log_every_steps': [1],
           'sampler_config.block_size': [1]},
          {'config.experiment.chain_length': [
              1000, 2000, 3000, 4000],
           'model_config.cfg_str': 's-gibbs,r-10k',
           'config.experiment.decay_rate': [0.05],
           'config.experiment.t_schedule': ['exp_decay'],
           'config.experiment.log_every_steps': [1],
           'sampler_config.block_size': [1]},
      ]
  ))
  return config