#TODO: update this
from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='normcut',
          graph_type='gap_rand',
          sampler='path_auxiliary',
          sweep=[
              {
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                      'r-VGG',
                      'r-RESNET',
                      'r-AlexNet',
                      'r-MNIST',
                      'r-BABELFISH',
                      'r-NMT',
                      'r-TRANSFORMER',
                      'r-TTS',
                  ],
                  'model_config.stype': ['quad', 'span'],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                      'r-VGG',
                      'r-RESNET',
                      'r-AlexNet',
                      'r-MNIST',
                      'r-BABELFISH',
                      'r-NMT',
                      'r-TRANSFORMER',
                      'r-TTS',
                  ],
                  'sampler_config.name': [
                      'dmala',
                      'path_auxiliary',
                      'gwg',
                      'dlmc',
                  ],
                  'model_config.stype': ['quad', 'span'],
                  'config.experiment.log_every_steps': [100],
                  'sampler_config.balancing_fn_type': [
                      'SQRT',
                  ],
              },
          ],
      )
  )
  return config