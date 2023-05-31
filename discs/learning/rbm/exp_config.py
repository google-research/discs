"""Config file for rbms."""

from discs.common import configs
from discs.common import train_configs
from ml_collections import config_dict


def get_config(rbm_config):
  """Get config for rbm learning/sampling."""
  dataset, vocab_size, num_hidden = rbm_config.split('-')
  vocab_size = int(vocab_size)
  num_hidden = int(num_hidden)
  config = configs.get_config()
  config.experiment.update(train_configs.get_common_train_config())
  config.experiment.learning_rate = 1e-3
  config.experiment.optimizer = 'adam'
  config.experiment.grad_norm = 0.0
  config.experiment.ema_decay = 0.0
  config.experiment.batch_size = 100
  config.experiment.shuffle_buffer_size = 50000
  config.experiment.pcd_steps = 100
  config.experiment.log_every_steps = 10
  config.experiment.save_every_steps = 100
  config.experiment.plot_every_steps = 10
  config.experiment.dataset = dataset
  config.model.num_hidden = num_hidden
  config.model.name = 'rbm'
  config.sampler = config_dict.ConfigDict(dict(
      name='blockgibbs',
      subclass='rbm'
  ))

  if dataset in ['mnist', 'fashion_mnist']:
    config.model.num_visible = 784
  else:
    raise ValueError('Unknown dataset %s' % dataset)
  config.model.num_categories = vocab_size
  config.experiment.rbm_config = rbm_config
  config.model.shape = (config.model.num_visible,)
  if config.model.num_hidden == 200:
    config.model.save_dir_name = 'rbm_lowtemp'
  elif config.model.num_hidden == 25:
    config.model.save_dir_name = 'rbm_hightemp'

  return config
