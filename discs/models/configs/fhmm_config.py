from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(2000,),
      num_categories=2,
      sigma=0.5,
      alpha=0.1,
      beta=0.8,
      L=100,
      K=10,
      name='fhmm',
  )
  model_config['save_dir_name'] = 'fhmm' 
  return config_dict.ConfigDict(model_config)
