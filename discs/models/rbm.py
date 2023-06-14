"""RBM Energy Function."""

import functools
import os
import pdb
import pickle
from typing import Any
from discs.models import deep_ebm
import flax
from flax import linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp
import ml_collections


class NNBinary(nn.Module):
  """Network of binary RBM."""

  num_visible: int
  num_hidden: int
  data_mean: Any = None

  def setup(self):
    if self.data_mean is None:
      bv_init = initializers.zeros
    else:
      data_mean = jnp.array(self.data_mean, dtype=jnp.float32)
      b_v = jnp.log(data_mean / (1.0 - data_mean))
      bv_init = lambda _, shape, dtype: jnp.reshape(b_v, shape).astype(dtype)
    self.b_v = self.param('b_v', bv_init, (self.num_visible,), jnp.float32)
    self.b_h = self.param(
        'b_h', initializers.zeros, (self.num_hidden,), jnp.float32
    )
    self.w = self.param(
        'w',
        initializers.glorot_uniform(),
        (self.num_visible, self.num_hidden),
        jnp.float32,
    )

  def __call__(self, v):
    """Un-normalized log-likelihood.

    Args:
      v: float tensor of size [batch_size, num_visible]

    Returns:
      ll: tensor of size [batch_size,]
    """
    sp = jnp.sum(jax.nn.softplus(v @ self.w + self.b_h), axis=-1)
    vt = jnp.sum(v * self.b_v, axis=-1)
    return sp + vt

  def step_h(self, rng, v):
    logits_h = v @ self.w + self.b_h
    return jax.random.bernoulli(key=rng, p=jax.nn.sigmoid(logits_h))

  def step_v(self, rng, h):
    logits_v = h @ self.w.T + self.b_v
    return jax.random.bernoulli(key=rng, p=jax.nn.sigmoid(logits_v))


class NNCategorical(nn.Module):
  """Network of categorical RBM."""

  num_visible: int
  num_hidden: int
  num_categories: int
  data_mean: Any = None

  def setup(self):
    if self.data_mean is None:
      bv_init = initializers.zeros
    else:
      data_mean = jnp.array(self.data_mean, dtype=jnp.float32)
      b_v = jnp.log(data_mean)
      bv_init = lambda _, shape, dtype: jnp.reshape(b_v, shape).astype(dtype)
    self.b_v = self.param(
        'b_v', bv_init, (self.num_visible, self.num_categories), jnp.float32
    )
    self.b_h = self.param(
        'b_h', initializers.zeros, (self.num_hidden,), jnp.float32
    )
    self.w = self.param(
        'w',
        initializers.normal(
            1.0 / jnp.sqrt(self.num_visible * self.num_categories + 2)
        ),
        (self.num_hidden, self.num_visible, self.num_categories),
        jnp.float32,
    )

  def __call__(self, v):
    sp = jnp.sum(
        jax.nn.softplus(
            jnp.sum(jnp.expand_dims(v, -3) * self.w, axis=[-1, -2]) + self.b_h
        ),
        -1,
    )
    vt = jnp.sum(v * self.b_v, axis=[-1, -2])
    return sp + vt


class RBM(deep_ebm.DeepEBM):
  """RBM."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.num_visible = config.num_visible
    self.num_hidden = config.num_hidden
    self.num_categories = config.num_categories
    self.net = None
    data_mean = None
    self.params = config.get('params', None)
    if self.params:
      self.params = flax.core.frozen_dict.unfreeze(self.params)
      data_mean = self.params['data_mean']
    self.init_dist = self.build_init_dist(data_mean)
    self.data_mean = data_mean
    self.shape = config.shape

  def get_init_samples(self, rng, num_samples: int):
    return self.init_dist(
        key=rng, shape=(num_samples, self.num_visible)
    ).astype(jnp.int32)

  def make_init_params(self, rng):
    if self.params:
      return self.params
    model_rng, sample_rng = jax.random.split(rng)
    v = self.get_init_samples(sample_rng, 1)
    return self.net.init({'params': model_rng}, v=v)['params']

  def forward(self, params, x):
    if self.num_categories != 2 and len(x.shape) - 1 == len(self.shape):
      x = jax.nn.one_hot(x, self.num_categories)
    return self.net.apply({'params': params}, v=x)

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)

    return loglikelihood, grad

  def step_h(self, params, rng, v):
    return self.net.apply(
        {'params': params}, rng=rng, v=v, method=self.net.step_h
    )

  def step_v(self, params, rng, h):
    return self.net.apply(
        {'params': params}, rng=rng, h=h, method=self.net.step_v
    )


class BinaryRBM(RBM):
  """RBM with binary observations."""

  def __init__(self, config: ml_collections.ConfigDict):
    super(BinaryRBM, self).__init__(config)
    self.net = NNBinary(
        num_visible=self.num_visible,
        num_hidden=self.num_hidden,
        data_mean=self.data_mean,
    )

  def build_init_dist(self, data_mean):
    if data_mean is None:
      return functools.partial(jax.random.bernoulli, p=0.5)
    else:
      return functools.partial(
          jax.random.bernoulli, p=jnp.array(data_mean, dtype=jnp.float32)
      )


class CategoricalRBM(RBM):
  """RBM with categorical observations."""

  def __init__(self, config: ml_collections.ConfigDict):
    super(CategoricalRBM, self).__init__(config)
    self.net = NNCategorical(
        num_visible=self.num_visible,
        num_hidden=self.num_hidden,
        num_categories=self.num_categories,
    )

  def build_init_dist(self, data_mean):
    if data_mean is None:
      return functools.partial(
          jax.random.categorical,
          logits=jnp.log(jnp.ones(self.num_visible, self.num_categories)),
      )
    else:
      logits = jnp.log(data_mean)
      return functools.partial(
          jax.random.categorical, logits=jnp.array(logits, dtype=jnp.float32)
      )


def build_model(config):
  config_model = config.model
  if config_model.num_categories == 2:
    return BinaryRBM(config_model)
  else:
    assert config_model.num_categories > 2
    return CategoricalRBM(config_model)
