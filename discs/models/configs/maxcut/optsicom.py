from discs.common import utils
from ml_collections import config_dict


def get_model_config(cfg_str):
  """Get config."""
  extra_cfg = utils.parse_cfg_str(cfg_str)
  num_nodes = num_edges = 0
  rand_type = extra_cfg['r']
  model_config = dict(
      num_models=1,
      max_num_nodes=num_nodes,
      max_num_edges=num_edges,
      num_categories=2,
      shape=(0,),
      rand_type=rand_type,
  )
  
  return config_dict.ConfigDict(model_config)
