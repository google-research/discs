from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='gibbswithgrad',
  )
  return config_dict.ConfigDict(model_config)
