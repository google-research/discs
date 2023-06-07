"""Resnet image models."""

import functools
from typing import Tuple
from discs.models import deep_ebm
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import pdb


def conv3x3(out_planes, stride=1):
  if stride < 0:
    mod = nn.ConvTranspose(features=out_planes, kernel_size=(3, 3),
                           strides=stride, padding=(1, 1), use_bias=True,
                           name='ConvTranspose')
  else:
    mod = nn.Conv(features=out_planes,
                  kernel_size=(3, 3),
                  strides=stride,
                  padding=(1, 1),
                  use_bias=True)
  return mod


class BasicBlock(nn.Module):
  """Basic building block of resnet."""

  planes: int
  stride: int = 1
  out_nonlin: bool = True
  expansion: int = 1

  @nn.compact
  def __call__(self, x):
    conv1 = conv3x3(self.planes, self.stride)
    out = nn.activation.swish(conv1(x))
    conv2 = conv3x3(self.planes)
    out = conv2(out)
    in_planes = x.shape[-1]
    if self.stride != 1 or in_planes != self.expansion * self.planes:
      if self.stride < 0:
        shortcut_conv = nn.ConvTranspose(
            features=self.expansion * self.planes, kernel_size=(1, 1),
            strides=-self.stride, use_bias=True, name='shortcut_conv_trans')
      else:
        shortcut_conv = nn.Conv(features=self.expansion * self.planes,
                                kernel_size=(1, 1), strides=self.stride,
                                use_bias=True, name='shortcut_conv')
      out_sc = shortcut_conv(x)
      out = out + out_sc
    else:
      out = out + x
    if self.out_nonlin:
      out = nn.activation.swish(out)
    return out


class ResnetBinary(nn.Module):
  """Network of binary resnet."""

  n_channels: int
  image_shape: Tuple[int, int] = (28, 28)

  @nn.compact
  def __call__(self, x):
    x = jnp.reshape(x, (x.shape[0],) + self.image_shape + (1,))
    x = nn.Conv(features=self.n_channels, kernel_size=(3, 3),
                strides=1, padding=(1, 1))(x)
    for _ in range(2):
      x = BasicBlock(planes=self.n_channels, stride=2)(x)
    for _ in range(6):
      x = BasicBlock(planes=self.n_channels, stride=1)(x)
    x = jnp.mean(x, axis=(1, 2))
    energy = nn.Dense(1)(x)
    return jnp.squeeze(energy, axis=-1)


class ImageEBM(deep_ebm.DeepEBM):
  """Common class of image EBMs."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.num_categories = config.num_categories
    self.net = None
    data_mean = config.get('data_mean', None)
    self.params = config.get('params', None)
    if self.params:
      self.params = flax.core.frozen_dict.unfreeze(self.params)
      data_mean = self.params['data_mean']
    self.init_dist = self.build_init_dist(data_mean)
    self.data_mean = data_mean
    if self.data_mean is not None:
      self.data_mean = jnp.array(self.data_mean, dtype=jnp.float32)
    self.shape = config.shape

  def get_init_samples(self, rng, num_samples: int):
    return self.init_dist(
        key=rng, shape=((num_samples,) + self.shape)
    ).astype(jnp.int32)

  def make_init_params(self, rng):
    if self.params:
      return self.params
    model_rng, sample_rng = jax.random.split(rng)
    x = self.get_init_samples(sample_rng, 1)
    return self.net.init({'params': model_rng}, x=x)['params']

  def forward(self, params, x):
    if self.num_categories != 2 and len(x.shape) - 1 == len(self.shape):
      x = jax.nn.one_hot(x, self.num_categories)
    energy = self.net.apply({'params': params}, x=x)
    if self.data_mean is not None:
      base_logprob = x * jnp.log(self.data_mean + 1e-20) + (
          1 - x) * jnp.log1p(-self.data_mean + 1e-20)
      base_energy = jnp.sum(base_logprob, axis=-1)
      energy = energy + base_energy
      if 'temperature' in params:  # for the use of AIS
        beta = params['temperature']
        energy = base_energy * (1 - beta) + energy * beta
    return energy

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad


class BinaryImageEBM(ImageEBM):
  """Image EBM with binary observations."""

  def __init__(self, config: ml_collections.ConfigDict):
    super(BinaryImageEBM, self).__init__(config)
    self.net = ResnetBinary(
        n_channels=config.n_channels,
        image_shape=config.image_shape,
    )

  def build_init_dist(self, data_mean):
    if data_mean is None:
      return functools.partial(jax.random.bernoulli, p=0.5)
    else:
      return functools.partial(
          jax.random.bernoulli, p=jnp.array(data_mean, dtype=jnp.float32)
      )


def build_model(config):
  config_model = config.model
  if config_model.num_categories == 2:
    return BinaryImageEBM(config_model)
  else:
    assert config_model.num_categories > 2
