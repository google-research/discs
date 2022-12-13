"""Locally Balanced Informed Sampler Class."""

from discs.samplers import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


class LocallyBalancedSampler(abstractsampler.AbstractSampler):
  """Locally Balanced Informed Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):

    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    # self.radius = config.radius
    self.balancing_fn_type = config.sampler.balancing_fn_type


  def make_init_state(self, rnd):
    """Returns expected number of flips."""
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
      state: the state of the sampler (changes in adaptive case to tune the
        proposal distribution).

    Returns:
      New sample.
    """

    def generate_new_samples(x):
      dim = np.prod(self.sample_shape)
      index_to_flip = jnp.identity(dim)
      index_to_flip_for_category = jnp.repeat(
          index_to_flip, self.num_categories - 1, axis=0)
      category_range = jnp.arange(1, self.num_categories)
      category_range = jnp.expand_dims(
          jnp.array(jnp.tile(category_range, dim)), 1)
      flipped = category_range * index_to_flip_for_category
      flipped_batch = jnp.vstack([flipped] * x.shape[0])
      x_flatten = x.reshape(x.shape[0], -1)
      y_flatten = jnp.repeat(x_flatten, dim * (self.num_categories - 1), axis=0)
      y_flatten = (y_flatten + flipped_batch) % self.num_categories
      y = y_flatten.reshape((y_flatten.shape[0],) + self.sample_shape)
      return y

    def select_new_samples(model, model_param, x, y, rnd_categorical, state):
      x_expanded = jnp.repeat(x, int(y.shape[0] / x.shape[0]), axis=0)
      t, new_state = get_ratio(model, model_param, x_expanded, y, state)
      locally_balance_proposal_dist = get_balancing_fn(t)
      proposal_dist_unnormalized = locally_balance_proposal_dist.reshape(
          x.shape[0], -1)
      proposal_dist = proposal_dist_unnormalized / jnp.sum(
          proposal_dist_unnormalized, axis=-1, keepdims=True)
      loglikelihood = jnp.log(proposal_dist)
      selected_y_index = random.categorical(
          rnd_categorical, loglikelihood, axis=1)
      dim = np.prod(self.sample_shape)
      selected_y_index_batch = (jnp.arange(x.shape[0]) * dim) + selected_y_index
      new_x = jnp.take(y, selected_y_index_batch, axis=0)
      return new_x, new_state

    def get_ratio(model, model_param, x, y, state):
      #TODO: define cutomized forward function in the model that only calculated modified regions.
      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)
      state = state.at[1].set(state[1] + 2)
      return jnp.exp(loglikelihood_y - loglikelihood_x), state

    def get_balancing_fn(t):
      #TODO: Enums
      if self.balancing_fn_type == 2:
        return t / (t + 1)
      elif self.balancing_fn_type == 3:  #and
        return jnp.where(t < 1, t, 1)
      elif self.balancing_fn_type == 4:  #or
        return jnp.where(t > 1, t, 1)
      return jnp.sqrt(t)

    num_log_like_calls = state[1]
    rnd_categorical, _ = random.split(rnd)
    del rnd
    y = generate_new_samples(x)
    new_x, new_state = select_new_samples(model, model_param, x, y,
                                          rnd_categorical, state)
    return new_x, new_state

def build_sampler(config):
  return LocallyBalancedSampler(config)