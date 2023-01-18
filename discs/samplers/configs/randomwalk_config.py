from ml_collections import config_dict


def get_config():
  model_config = dict(
      adaptive=True,
      target_acceptance_rate=0.237,
      name='randomwalk',
  )
  return config_dict.ConfigDict(model_config)
