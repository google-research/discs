"""Main Config Structure."""

from ml_collections import config_dict


def get_common_train_config():
  train_config = dict(
      learning_rate=1e-3,
      lr_schedule='constant',
      warmup_frac=0.05,
      total_train_steps=1000000,
      optimizer='adamw',
      grad_norm=5.0,
      weight_decay=0.0,
      ema_decay=0.9999,
  )
  return config_dict.ConfigDict(train_config)


def get_config():
  """Get common config sketch."""
  general_config = dict(
      model=dict(
          name='',
      ),
      sampler=dict(
          name='',
      ),
      experiment=dict(
          name='Text_Infilling_Experiment',
          evaluator='ess_eval',
          num_models=1,
          batch_size=1,
          chain_length=1000,
          ess_ratio=0.5,
          run_parallel=True,
          get_additional_metrics=True,
          t_schedule='constant',
          init_temperature=1.0,
          window_size=10,
          window_stride=10,
          shuffle_buffer_size=0,
          log_every_steps=1,
          plot_every_steps=10,
          save_root='.',
          fig_folder='',
          save_every_steps=10000,
      ),
  )
  return config_dict.ConfigDict(general_config)
