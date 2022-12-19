from ml_collections import config_dict


def get_config():
  model_config = dict(
      name='gibbswithgrad',
      adaptive=False,
      target_acceptance_rate=0.574,
      balancing_fn_type=0,
  )
  return config_dict.ConfigDict(model_config)