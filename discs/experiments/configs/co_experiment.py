from ml_collections import config_dict


def get_co_default_config():
  """Get combinatorial default configs."""
  exp_config = config_dict.ConfigDict()
  exp_config.evaluator = 'co_eval'
  exp_config.use_tqdm = True
  exp_config.batch_size = 1
  exp_config.num_models = 1
  exp_config.chain_length = 50000
  exp_config.t_schedule = 'linear'
  exp_config.init_temperature = 1.0
  exp_config.decay_rate = 0.005
  exp_config.final_temperature = 0.05
  exp_config.log_every_steps = 100
  exp_config.temp0_steps = 0
  exp_config.name = 'CO_Experiment'
  return exp_config
