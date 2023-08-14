from ml_collections import config_dict


def get_model_config(cfg_str):
  """Get config for RB graphs."""
  _ = cfg_str
  num_nodes = 247
  num_edges = 12174
  num_instances = 196

  model_config = dict(
      num_models=200,
      max_num_nodes=num_nodes,
      max_num_edges=num_edges,
      num_instances=num_instances,
      num_categories=2,
      shape=(0,),
      rand_type='',
      penalty=1.0,
  )

  return config_dict.ConfigDict(model_config)
