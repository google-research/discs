"""Main Config Structure"""
from ml_collections import config_dict


def get_config():
  general_config = dict(
      model=dict(
          name='',
      ),
      sampler=dict(
          name='',
      ),
      experiment=dict(
          run_parallel=False,
          batch_size=100,
          chain_length=100000,
          chain_burnin_length=900,
          window_size=10,
          window_stride=10,
          ess_ratio = 0.5,
      ),
  )
  return config_dict.ConfigDict(general_config)
