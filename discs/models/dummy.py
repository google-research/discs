import importlib
from ml_collections import config_dict
from ml_collections import config_flags
from discs.models.configs import maxcut_config
import pdb


def get_general_config():
  general_config = dict(
      model=dict(
          name='',
  ))
  return config_dict.ConfigDict(general_config)

def get_model_config():
  # getting the base config of maxcut model.
  config = get_general_config()
  config.model.update(maxcut_config.get_config())
  if config.model.get('graph_type', None):
    graph_config = importlib.import_module('discs.models.configs.%s.%s'% (config.model.name, config.model.graph_type))
    config.model.update(graph_config.get_model_config(config.model.cfg_str))
  return config

#pdb.set_trace()
# config
config =get_model_config()
# model
model_mod = importlib.import_module('discs.models.%s' % config.model.name)
model = model_mod.build_model(config)
print("Endddddddddddddddddd")