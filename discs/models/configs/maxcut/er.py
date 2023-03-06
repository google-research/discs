"""Config for maxcut ba dataset."""

from discs.common import configs
from ml_collections import config_dict
from sco.experiments import default_configs


def get_config(cfg_str):
  """Get config."""
  extra_cfg = default_configs.parse_cfg_str(cfg_str)
  num_nodes = num_edges = 0
  rand_type = extra_cfg['r']
  if rand_type.endswith('1024-1100'):
    num_nodes = 1100
    num_edges = 91239
  elif rand_type.endswith('512-600'):
    num_nodes = 600
    num_edges = 27132
  elif rand_type.endswith('256-300'):
    num_nodes = 300
    num_edges = 6839

  model_config = dict(
      name='maxcut',
      max_num_nodes=num_nodes,
      max_num_edges=num_edges,
      num_instances=1000,
      num_categories=2,
      shape=(0,),
      rand_type=rand_type,
      data_root='',
      graph_type='maxcut-er',
  )
  config.experiment.update(default_configs.get_exp_config())
  config.sampler = default_configs.get_sampler_config(extra_cfg['s'])

  config.experiment.batch_size = 1024
  if num_nodes >= 300:
    config.experiment.batch_size = 128
  config.experiment.samples_per_instance = 16
  config.experiment.t_schedule = 'linear'
  config.experiment.chain_length = 10000
  config.experiment.log_every_steps = 1
  config.experiment.init_temperature = 1
  config.experiment.decay_rate = 0.1
  config.experiment.final_temperature = 0.5
  config.sampler.approx_with_grad = False
  return config_dict.ConfigDict(model_config)