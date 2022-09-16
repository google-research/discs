"""Gibbs Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
from itertools import product
import jax.numpy as jnp
import jax
import ml_collections
import pdb


class BlockGibbsSampler(abstractsampler.AbstractSampler):
  """Gibbs Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_dimension = config.sample_dimension
    self.num_categories = config.num_categories
    self.random_order = config.random_order
    self.block_size = config.block_size
    self.block_size = max(min(self.sample_dimension, self.block_size), 1)

  def make_init_state(self, rnd):
    return 1

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

    def generate_new_samples(indices, flip_index_start, x):

      y = jnp.repeat(x, self.num_categories**self.block_size, axis=0)
      categories_iter = jnp.array(
          list(product(range(self.num_categories), repeat=self.block_size)))
      category_iteration = jnp.vstack([categories_iter] * x.shape[0])
      indices_to_flip = indices[flip_index_start:flip_index_start +
                                self.block_size]
      category_iteration = jax.lax.dynamic_slice(
          category_iteration, (0, 0),
          (category_iteration.shape[0], indices_to_flip.shape[0]))
      y = y.at[:, indices_to_flip].set(category_iteration)
      return y

    def select_new_samples(model, model_param, x, y, rnd_categorical):
      loglikelihood = model.forward(model_param, y)
      loglikelihood = loglikelihood.reshape(
          -1, self.num_categories**self.block_size)
      new_x_dim = random.categorical(
          rnd_categorical, loglikelihood, axis=1).reshape(-1)
      selected_index = (jnp.arange(x.shape[0]) *
                        (self.num_categories**self.block_size)) + new_x_dim
      x = jnp.take(y, selected_index, axis=0)
      return x

    # iterative conditional
    rnd_shuffle, rnd_categorical = random.split(rnd)
    del rnd
    indices = jnp.arange(x.shape[-1])
    if self.random_order:
      indices = random.shuffle(rnd_shuffle, indices, axis=0)
    for flip_index_start in range(0, len(indices), self.block_size):
      y = generate_new_samples(indices, flip_index_start, x)
      x = select_new_samples(model, model_param, x, y, rnd_categorical)
    new_x = x
    new_state = state
    
    return new_x, new_state
