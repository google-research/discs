"""Locally Balanced Informed Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections
import pdb
import math

class LocallyBalancedSampler(abstractsampler.AbstractSampler):
  """Locally Balanced Informed Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    if isinstance(config.sample_shape, int):
      self.sample_shape = (config.sample_shape,)
    else:
      self.sample_shape = config.sample_shape
    self.num_categories = config.num_categories
    # self.radius = config.radius
    self.balancing_fn_type = config.balancing_fn_type

  def make_init_state(self, rnd):
    """Returns expected number of flips."""
    return 1  #random.uniform(rnd, shape=(1, 1), minval=1, maxval=self.sample_shape).at[0, 0].get()

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
      dim = math.prod(self.sample_shape)
      index_to_flip = jnp.identity(dim)
      index_to_flip_for_category = jnp.repeat(
          index_to_flip, self.num_categories - 1, axis=0)
      category_range = jnp.arange(1, self.num_categories)
      category_range = jnp.expand_dims(
          jnp.array(jnp.tile(category_range, dim)), 1)
      flipped = category_range * index_to_flip_for_category
      flipped_batch = jnp.vstack([flipped]*x.shape[0])
      x_flatten = x.reshape(x.shape[0], -1)
      y_flatten = jnp.repeat(x_flatten, dim*(self.num_categories - 1), axis=0)
      y_flatten = (y_flatten + flipped_batch) % self.num_categories
      y = y_flatten.reshape( (y_flatten.shape[0],) + self.sample_shape)
      return y

    def select_new_samples(model, model_param, x, y, rnd_categorical):
      x_expanded = jnp.repeat(x, int(y.shape[0] / x.shape[0]), axis=0)
      t = get_ratio(model, model_param, x_expanded, y)
      locally_balance_proposal_dist = get_balancing_fn(t)
      loglikelihood = locally_balance_proposal_dist.reshape(x.shape[0], -1)
      selected_y_index = random.categorical(
          rnd_categorical, loglikelihood, axis=1)
      dim = math.prod(self.sample_shape)
      selected_y_index_batch = (jnp.arange(x.shape[0]) * dim) + selected_y_index
      new_x = jnp.take(y, selected_y_index_batch, axis=0)
      return new_x

    def get_ratio(model, model_param, x, y):
      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)
      return jnp.exp(loglikelihood_y - loglikelihood_x)

    def get_balancing_fn(t):
      if self.balancing_fn_type == 2:
        return t / (t + 1)
      elif self.balancing_fn_type == 3: #and
        return jnp.where(t < 1, t, 1)
      elif self.balancing_fn_type == 4: #or
        return jnp.where(t > 1, t, 1)
      return jnp.sqrt(t)
    
    rnd_categorical, _ = random.split(rnd)
    del rnd
    y = generate_new_samples(x)
    new_x = select_new_samples(model, model_param, x, y, rnd_categorical)
    new_state = state

    return new_x, new_state
