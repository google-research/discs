"""Config file for categorical models."""
from ml_collections import config_dict

def get_config():
  model_config = dict(
      shape=(2000,),
      num_categories=4,
      init_sigma=1.5,
      name='categorical',
  )
  model_config['save_dir_name'] = 'categorical_' + str(
      model_config['num_categories']
  )
  return config_dict.ConfigDict(model_config)
