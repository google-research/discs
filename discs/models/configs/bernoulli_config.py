from ml_collections import config_dict


def get_config():
  model_config = dict(
    shape=(10000, ),
    num_categories=2,
    init_sigma=1.0,
    name='bernoulli',
  )
  return config_dict.ConfigDict(model_config)
