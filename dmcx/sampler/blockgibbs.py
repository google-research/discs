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
      state: the state of the sampler (changes in adaptive case to tune the
        proposal distribution).

    Returns:
      New sample.
    """

    def get_new_sample(rnd_new_sample, x, model, model_param):
      """Proposal distribution to sample the next state.

      Args:
        rnd_new_sample: key for binary mask and random flip.
        x: current sample.
        model: target distribution.
        model_param: target distribution parameters used for loglikelihood
          calulation.

      Returns:
        New sample.
      """

      rnd_shuffle, rnd_categorical = random.split(rnd_new_sample)
      del rnd_new_sample
      indices = jnp.arange(x.shape[-1])
      if self.random_order:
        indices = random.shuffle(rnd_shuffle, indices, axis=0)
      for dim in range(0, len(indices), self.block_size):
        x_expanded = jnp.repeat(x, self.num_categories**self.block_size, axis=0)
        categories_iter = jnp.array(
            list(product(range(self.num_categories), repeat=self.block_size)))
        category_iteration = jnp.vstack([categories_iter] * x.shape[0])
        indices_to_flip = indices[dim:dim + self.block_size]
        category_iteration = jax.lax.dynamic_slice(
            category_iteration, (0, 0),
            (category_iteration.shape[0], indices_to_flip.shape[0]))
        x_expanded = x_expanded.at[:, indices_to_flip].set(category_iteration)
        loglikelihood = model.forward(model_param, x_expanded)
        loglikelihood = loglikelihood.reshape(
            -1, self.num_categories**self.block_size)
        new_x_dim = random.categorical(
            rnd_categorical, loglikelihood, axis=1).reshape(-1)
        selected_index = (jnp.arange(x.shape[0]) *
                          (self.num_categories**self.block_size)) + new_x_dim
        x = jnp.take(x_expanded, selected_index, axis=0)
      return x

    new_x = get_new_sample(rnd, x, model, model_param)
    new_state = state
    return new_x, new_state
