from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(2000,),
      num_categories=4,
      init_sigma=1.5,
      name='categorical',
  )
  return config_dict.ConfigDict(model_config)
