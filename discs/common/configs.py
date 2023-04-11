"""Main Config Structure."""

from ml_collections import config_dict


def get_config():
  """Get common config sketch."""
  general_config = dict(
      model=dict(
          name='',
          data_root='sco',
      ),
      sampler=dict(
          name='',
      ),
      experiment=dict(
          name='Sampling_Experiment',
          evaluator='ess_eval',
          num_models=1,
          batch_size=100,
          chain_length=100000,
          ess_ratio=0.5,
          run_parallel=True,
          get_additional_metrics=True,
          t_schedule='constant',
          decay_rate=0.01,
          init_temperature=1.0,
          window_size=10,
          window_stride=10,
          shuffle_buffer_size=0,
          log_every_steps=1,
          plot_every_steps=10,
          save_root='./discs/results',
          fig_folder='',
          save_every_steps=10000,
          use_tqdm=False,
      ),
  )    
  return config_dict.ConfigDict(general_config)
