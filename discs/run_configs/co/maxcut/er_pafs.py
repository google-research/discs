from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='maxcut',
          sampler='path_auxiliary',
          graph_type='er',
          sweep=[
              {
                  'config.experiment.chain_length': [50000],
                  'sampler_config.name': [
                      'path_auxiliary',
                  ],
                  'sampler_config.approx_with_grad': [False, True],
                  'model_config.cfg_str': [
                      'r-er-0.15-n-1024-1100',
                      'r-er-0.15-n-512-600',
                      'r-er-0.15-n-256-300',
                  ],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
          ],
      )
  )
  return config