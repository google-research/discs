"""Gibbs Sampler Class."""

from discs.samplers import abstractsampler
from jax import random
from itertools import product
import jax.numpy as jnp
import jax
import ml_collections
import numpy as np
import pdb


class BlockGibbsSampler(abstractsampler.AbstractSampler):
  """Gibbs Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.random_order = config.sampler.random_order
    self.block_size = config.sampler.block_size

  def make_init_state(self, rnd):
    num_log_like_calls = 0
    return jnp.array([1, num_log_like_calls])

  def step(self, model, rnd, x, model_param, state):
    """Given the current sample, returns the next sample of the chain.

    Args:
      model: target distribution.
      rnd: random key generator for JAX.
      x: current sample.
      model_param: target distribution parameters used for loglikelihood
        calulation.
      state: the state of the sampler.

    Returns:
      New sample.
    """

    def generate_new_samples(indices_to_flip, x):
      x_flatten = x.reshape(x.shape[0], -1)
      y_flatten = jnp.repeat(
          x_flatten, self.num_categories**self.block_size, axis=0
      )
      categories_iter = jnp.array(
          list(product(range(self.num_categories), repeat=self.block_size))
      )
      category_iteration = jnp.vstack([categories_iter] * x.shape[0])
      category_iteration = jax.lax.dynamic_slice(
          category_iteration,
          (0, 0),
          (category_iteration.shape[0], indices_to_flip.shape[0]),
      )
      y_flatten = y_flatten.at[:, indices_to_flip].set(category_iteration)
      y = y_flatten.reshape((y_flatten.shape[0],) + self.sample_shape)
      return y

    def select_new_samples(model, model_param, x, y, rnd_categorical, state):
      loglikelihood = model.forward(model_param, y)
      state = state.at[1].set(state[1] + 1)
      loglikelihood = loglikelihood.reshape(
          -1, self.num_categories**self.block_size
      )
      new_x_dim = random.categorical(
          rnd_categorical, loglikelihood, axis=1
      ).reshape(-1)
      selected_index = (
          jnp.arange(x.shape[0]) * (self.num_categories**self.block_size)
      ) + new_x_dim
      x = jnp.take(y, selected_index, axis=0)
      return x, state

    # iterative conditional
    rnd_shuffle, rnd_categorical = random.split(rnd)
    del rnd
    dim = np.prod(self.sample_shape)
    indices = jnp.arange(dim)
    if self.random_order:
      indices = random.permutation(
          rnd_shuffle, indices, axis=0, independent=True
      )
    # TODO: use jax.lax.scan or jax.lax.fori_loop
    for flip_index_start in range(0, len(indices), self.block_size):
      indices_to_flip = indices[
          flip_index_start : flip_index_start + self.block_size
      ]
      y = generate_new_samples(indices_to_flip, x)
      x, new_state = select_new_samples(
          model, model_param, x, y, rnd_categorical, state
      )
    new_x = x

    return new_x, new_state


def build_sampler(config):
  return BlockGibbsSampler(config)
