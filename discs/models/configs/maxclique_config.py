"""Config file for graph models for maxclique problem."""
from ml_collections import config_dict

def get_config():
  model_config = config_dict.ConfigDict(
      dict(
          name='maxclique',
          graph_type='rb',
          cfg_str='_',
          data_root='./sco/',
      )
  )
  model_config['save_dir_name'] = model_config['name']
  return model_config
