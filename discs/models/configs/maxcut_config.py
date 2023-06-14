"""Config file for graph models for maxcut problem."""
from ml_collections import config_dict

def get_config():
  model_config = config_dict.ConfigDict(
      dict(
          name='maxcut',
          graph_type='ba',
          cfg_str='r-ba-4-n-16-20',
          data_root='./sco/',
      )
  )
  model_config['save_dir_name'] = model_config['name']
  return model_config
