"""Default experiment configs for combinatorial optimization problems."""
from ml_collections import config_dict


def get_co_default_config():
  """Get combinatorial default configs."""
  exp_config = config_dict.ConfigDict()
  exp_config.evaluator = 'co_eval'
  exp_config.name = 'CO_Experiment'
  exp_config.co_opt_prob = True
  return exp_config
