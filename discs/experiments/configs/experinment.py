from ml_collections import config_dict


def get_config():
  """Get common config sketch."""
  experiment_config = dict(
      save_root='.',
      fig_folder='',
      run_parallel=False,
      batch_size=100,
      chain_length=100000,
      window_size=10,
      window_stride=10,
      shuffle_buffer_size=0,
      log_every_steps=1,
      plot_every_steps=10,
      save_every_steps=100,
      ess_ratio=0.5,
      evaluator='ess_eval',  # ess_eval or co_eval
  )
  return config_dict.ConfigDict(experiment_config)


def get_co_default_config():
  """Get combinatorial default configs."""
  exp_config = config_dict.ConfigDict()
  exp_config.use_tqdm = True
  exp_config.batch_size = 1
  exp_config.num_graphs = 1
  exp_config.chain_length = 50000
  exp_config.t_schedule = 'linear'
  exp_config.init_temperature = 1.0
  exp_config.decay_rate = 0.005
  exp_config.final_temperature = 0.05
  exp_config.log_every_steps = 100
  exp_config.temp0_steps = 0
  return exp_config
