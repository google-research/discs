"""Config file for graph models for normcut problem."""
from ml_collections import config_dict

def get_config():
  model_config = config_dict.ConfigDict(
      dict(
          name='normcut',
          graph_type='nets',
          cfg_str='r-INCEPTION',
          data_root='./sco/',
          stype='quad',
      )
  )
  model_config['save_dir_name'] = model_config['name']
  return model_config
