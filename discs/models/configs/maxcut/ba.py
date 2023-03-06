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
    num_edges = 4384
  elif rand_type.endswith('512-600'):
    num_nodes = 600
    num_edges = 2384
  elif rand_type.endswith('256-300'):
    num_nodes = 300
    num_edges = 1184

  model_config = dict(
      name='maxcut',
      max_num_nodes=num_nodes,
      max_num_edges=num_edges,
      num_instances=1000,
      num_categories=2,
      shape=(0,),
      rand_type=rand_type,
      data_root='',
      graph_type='maxcut-ba',
  )
  config.experiment.update(default_configs.get_exp_config())

  config.experiment.batch_size = 1024
  config.experiment.samples_per_instance = 16
  config.experiment.t_schedule = 'exp_decay'
  config.experiment.chain_length = 10000
  config.experiment.log_every_steps = 10
  config.experiment.init_temperature = 0.5
  config.experiment.decay_rate = 0.05
  config.experiment.final_temperature = 0.000001
  
  return config_dict.ConfigDict(model_config)