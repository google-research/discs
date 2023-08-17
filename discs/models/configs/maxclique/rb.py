from ml_collections import config_dict


def get_model_config(cfg_str):
  """Get config for RB graphs."""
  _ = cfg_str
  num_nodes = 475
  num_edges = 90585
  num_instances = 500

  model_config = dict(
      num_models=512,
      name='maxclique',
      max_num_nodes=num_nodes,
      max_num_edges=num_edges,
      num_instances=num_instances,
      num_categories=2,
      shape=(0,),
      rand_type='',
      penalty=1.0,
  )

  return config_dict.ConfigDict(model_config)
