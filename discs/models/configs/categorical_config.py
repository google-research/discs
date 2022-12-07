from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(2, 2),
      num_categories=3,
      init_sigma=1.0,
      name='categorical',
  )
  return config_dict.ConfigDict(model_config)