from ml_collections import config_dict


def get_config():
  model_config = dict (
      block_size=2,
      name='blockgibbs_update',
  )
  return config_dict.ConfigDict(model_config)
