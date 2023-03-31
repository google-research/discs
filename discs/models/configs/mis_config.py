"""Config file for graph models for mis problem."""

import importlib
from ml_collections import config_dict


def get_config():
  model_config = config_dict.ConfigDict(
      dict(
          name='mis',
          graph_type='ertest',
          data_root='er_800_test',
          cfg_str='r-800',
      )
  )
  model_config['save_dir_name'] = model_config['name']
  graph_config = importlib.import_module(
      'discs.models.configs.mis.%s' % model_config['graph_type']
  )
  model_config.update(graph_config.get_model_config(model_config['cfg_str']))
  return model_config
