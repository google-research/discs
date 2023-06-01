#TODO:  'r-TRANSFORMER' is not working
from ml_collections import config_dict


def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='normcut',
          graph_type='nets',
          sampler='path_auxiliary',
          sweep=[
              {
                  'config.experiment.chain_length': [1000000],
                  'sampler_config.name': [
                      'randomwalk',
                      'blockgibbs',
                      'hammingball',
                  ],
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                      'r-VGG',
                      'r-RESNET',
                      'r-ALEXNET',
                      'r-MNIST',
                      'r-BABELFISH',
                      'r-NMT',
                      'r-TTS',
                  ],
                  'model_config.stype': ['quad', 'span'],
                  'config.experiment.log_every_steps': [100],
              },
              {
                  'config.experiment.chain_length': [1000000],
                  'model_config.cfg_str': [
                      'r-INCEPTION',
                      'r-VGG',
                      'r-RESNET',
                      'r-ALEXNET',
                      'r-MNIST',
                      'r-BABELFISH',
                      'r-NMT',
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