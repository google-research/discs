"""Ising Energy Function."""

import os
from os.path import exists

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
from dmcx.model import abstractmodel
from tqdm import tqdm
import dmcx.sampler.blockgibbs as blockgibbs_sampler
import pdb
import jax
import jax.numpy as jnp
import ml_collections
import pickle


class Ising(abstractmodel.AbstractModel):
  """Ising Distribution with Cyclic 2D Lattice."""

  def __init__(self, config: ml_collections.ConfigDict):

    if isinstance(config.shape, int):
      self.shape = (config.shape, config.shape)
    else:
      self.shape = config.shape
    self.lambdaa = config.lambdaa
    self.external_field_type = config.external_field_type
    self.init_sigma = config.init_sigma
   
  def make_init_params(self, rnd):
    # connectivity strength
    params_weight_h = self.lambdaa * jnp.ones(self.shape)
    params_weight_v = self.lambdaa * jnp.ones(self.shape)

    # external force
    if self.external_field_type == 1:
      params_b = jax.random.normal(rnd, shape=self.shape) * self.init_sigma
    else:
      params_b = jnp.zeros(shape=self.shape)
    return jnp.array([params_b, params_weight_h, params_weight_v])

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.randint(
        rnd,
        shape=(num_samples,) + self.shape,
        minval=0,
        maxval=2,
        dtype=jnp.int32)
    return x0

  def forward(self, params, x):

    x = 2 * x - 1
    w_b = params[0]
    w_h = params[1][:, :-1]
    w_v = params[2][:-1, :]
    sum_neighbors = jnp.zeros((x.shape[0],) + self.shape)
    sum_neighbors = sum_neighbors.at[:, :, :-1].set(
        sum_neighbors[:, :, :-1] + x[:, :, :-1] * x[:, :, 1:] * w_h)  # right
    sum_neighbors = sum_neighbors.at[:, :, 1:].set(
        sum_neighbors[:, :, 1:] + x[:, :, 1:] * x[:, :, :-1] * w_h)  # left
    sum_neighbors = sum_neighbors.at[:, :-1, :].set(
        sum_neighbors[:, :-1, :] + x[:, :-1, :] * x[:, 1:, :] * w_v)  # down
    sum_neighbors = sum_neighbors.at[:, 1:, :].set(
        sum_neighbors[:, 1:, :] + x[:, 1:, :] * x[:, :-1, :] * w_v)  # up
    biases = w_b * x
    loglikelihood = sum_neighbors + biases
    loglikelihood = jnp.sum((loglikelihood).reshape(x.shape[0], -1), axis=-1)
    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad
