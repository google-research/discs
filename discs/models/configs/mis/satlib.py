"""Config for satlib dataset."""

from discs.common import configs
from ml_collections import config_dict


def get_config(cfg_str):
  """Get config for er benchmark graphs."""
  num_nodes = 1347
  num_edges = 5978
  num_instances = 500

  model_config = dict(
      name='mis',
      max_num_nodes=num_nodes,
      max_num_edges=num_edges,
      num_instances=num_instances,
      num_categories=2,
      shape=(0,),
      rand_type='',
      penalty=1.0001,
      data_root='',
      graph_type='satlib',
  )
  return config_dict.ConfigDict(model_config)
