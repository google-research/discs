"""Config file for rbms."""

from discs.common import configs
from discs.models.configs import rbm_config as model_config
from ml_collections import config_dict


def get_config(rbm_config):
  """Get config for rbm learning/sampling."""
  config = configs.get_config()
  config.model = model_config.get_config()
  dataset = config.model.dataset
  vocab_size = config.model.num_categories
  num_hidden = config.model.num_hidden
  vocab_size = int(vocab_size)
  num_hidden = int(num_hidden)
  config.experiment.update(configs.get_common_train_config())
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
  config.model.train = True
  config.sampler = config_dict.ConfigDict(dict(
      name='blockgibbs',
      subclass='rbm'
  ))

  if dataset not in ['mnist', 'fashion_mnist']:
    raise ValueError('Unknown dataset %s' % dataset)
  config.experiment.rbm_config = rbm_config
  return config
