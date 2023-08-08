"""Config for er-test dataset."""

from ml_collections import config_dict


def get_model_config(cfg_str):
  _ = cfg_str
  model_config = dict(
      num_models=5,
      max_num_nodes=1000,
      max_num_edges=50369,
      num_instances=5,
      num_categories=3,
      shape=(0,),
      rand_type='test-1000',
      penalty=1.0,
      tolerance=0.03,
  )
  return config_dict.ConfigDict(model_config)
