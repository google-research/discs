from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(2, 2),
      num_categories=2,
      lambdaa=0.1,
      external_field_type=1,
      mu=0.5,
      init_sigma=1.0,
      name='ising',
  )
  return config_dict.ConfigDict(model_config)