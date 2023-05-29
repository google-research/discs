"""Replay buffer."""

import functools
import numpy as np


class ReplayBuffer(object):
  """Replay buffer."""

  def __init__(self, config, x_train):
    exp_config = config.experiment
    self.data_shape = x_train.shape[1:]
    self.buffer_size = exp_config.buffer_size
    self.is_categorical = config.model.num_categories > 2

    eps = 1e-2
    init_mean = np.mean(x_train, axis=0) * (1. - 2 * eps) + eps

    if self.is_categorical:
      assert config.model.num_categories == 256  # image
      pass
    else:
      assert exp_config.buffer_init == 'mean'
      self.init_dist = functools.partial(
          np.random.binomial, n=1, p=init_mean)
      self.buffer = self.init_dist(size=(self.buffer_size,) + self.data_shape)
      self.reinit_dist = functools.partial(
          np.random.binomial, n=1, p=exp_config.reinit_freq)
    self.all_inds = list(range(self.buffer_size))

  def sample(self, batch_size):
    """Samle batch_size samples from buffer."""
    buffer_inds = sorted(
        np.random.choice(self.all_inds, batch_size, replace=False))
    x_buffer = self.buffer[buffer_inds]
    if self.is_categorical:
      x_fake = x_buffer
    else:
      x_reinit = self.init_dist(size=(batch_size,) + self.data_shape)
      x_reinit = x_reinit.astype(x_buffer.dtype)
      reinit = self.reinit_dist(size=[batch_size] + [1] * len(self.data_shape))
      reinit = reinit.astype(x_buffer.dtype)
      x_fake = x_reinit * reinit + x_buffer * (1 - reinit)
    return x_fake, buffer_inds

  def update(self, x, buffer_inds):
    self.buffer[buffer_inds] = x
