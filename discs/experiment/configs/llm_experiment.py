from ml_collections import config_dict


def get_llm_default_config():
  """Get combinatorial default configs."""
  exp_config = config_dict.ConfigDict()
  exp_config.evaluator = 'llm_eval'
  exp_config.name = 'Text_Infilling_Experiment'
  exp_config.batch_size = 1
  exp_config.chain_length = 50
  exp_config.max_n = 4
  exp_config.num_same_resample = 25
  exp_config.run_parallel = False
  return exp_config
