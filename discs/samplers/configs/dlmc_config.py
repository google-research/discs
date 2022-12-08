from ml_collections import config_dict


def get_config():
  model_config = dict(
      adaptive=True,
      target_acceptance_rate=0.574,
      name='dlmc',
      fixed_log_tau=0.,
      balancing_fn_type=1,
  )
  return config_dict.ConfigDict(model_config)
