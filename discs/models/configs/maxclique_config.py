"""Config file for graph models for maxcut problem."""

import importlib
from ml_collections import config_dict

def get_config():
  model_config = config_dict.ConfigDict(
      dict(
          name='maxclique',
          graph_type='rb',
      )
  )
  model_config['save_dir_name'] = model_config['name']
  graph_config = importlib.import_module(
      'discs.models.configs.maxclique.%s' % model_config['graph_type']
  )
  model_config.update(graph_config.get_model_config())
  return model_config
