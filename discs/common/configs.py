from ml_collections import config_dict

def get_config():
  general_config = dict(
    model=dict(
        name='',
    ),
    sampler=dict(
        name='',
    ),
    experiment=dict(),
  )
  return config_dict.ConfigDict(general_config)
