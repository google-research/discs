"""Config file for graph models for mis problem."""
from ml_collections import config_dict

def get_config():
  model_config = config_dict.ConfigDict(
      dict(
          name='mis',
          graph_type='ertest',
          cfg_str='r-800',
          data_root='./sco/',
          penalty=1.001,
      )
  )
  model_config['save_dir_name'] = model_config['name']
  return model_config
