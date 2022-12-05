from ml_collections import config_dict


def get_config():
  model_config = dict(
      random_order=False,
      block_size=3,
      name='blockgibbs',
  )
  return config_dict.ConfigDict(model_config)
