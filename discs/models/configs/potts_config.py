from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(10, 10),
      lambdaa=0.1,
      init_sigma=1.0,
      num_categories=2,
      name='potts',
  )
  return config_dict.ConfigDict(model_config)