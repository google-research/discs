"""Config file for bernoulli models."""

from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(10000,),
      num_categories=2,
      name='maxcut',
      graph_type='ba',
      cfg_str='r-ba-4-n-32-40'
  )
  model_config['save_dir_name']=cfg_str

  return config_dict.ConfigDict(model_config)
