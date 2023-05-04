from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='hammingball',
      block_size=10,
      hamming_dist=1
  )
  return config_dict.ConfigDict(model_config)
