"""Default experiment configs for language model."""
from ml_collections import config_dict


def get_config():
  """Get config."""
  exp_config = dict(
      experiment=dict(
          evaluator='lm_eval',
          name='Text_Infilling_Experiment',
          batch_size=1,
          chain_length=50,
          max_n_grams=4,
          num_same_resample=25,
          topk_num=5,
          use_topk=False,
          run_parallel=False,
          save_root='',
      )
  )
  return config_dict.ConfigDict(exp_config)
