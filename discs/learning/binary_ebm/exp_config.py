"""Config file for binary image ebms."""

from discs.common import configs
from discs.common import train_configs
from discs.samplers.configs import dlmc_config


def get_config(img_config):
  """Get config for image ebm learning/sampling."""

  dataset, num_channels = img_config.split('-')
  config = configs.get_config()
  config.experiment.update(train_configs.get_common_train_config())
  config.experiment.img_config = img_config
  config.experiment.total_train_steps = 2
  config.experiment.dataset = dataset
  config.experiment.batch_size = 100
  config.experiment.buffer_size = 10000
  config.experiment.reinit_freq = 0.0
  config.experiment.buffer_init = 'mean'
  config.experiment.optimizer = 'adam'
  config.experiment.p_control = 0.0
  config.experiment.energy_l2 = 0.0
  config.experiment.pcd_steps = 1
  config.model.name = 'resnet'
  config.model.num_categories = 2
  config.model.n_channels = int(num_channels)

  config.sampler = dlmc_config.get_config()

  if dataset in ['dynamic_mnist', 'static_mnist', 'omniglot', 'caltech']:
    config.model.image_shape = (28, 28)
    config.model.shape = (784,)

  return config
