"""Main Train Config Structure."""

from ml_collections import config_dict


def get_common_train_config():
  train_config = dict(
      learning_rate=1e-3,
      data_dir='',
      lr_schedule='constant',
      warmup_frac=0.05,
      total_train_steps=1000000,
      optimizer='adamw',
      grad_norm=5.0,
      weight_decay=0.0,
      ema_decay=0.9999,
  )
  return config_dict.ConfigDict(train_config)
