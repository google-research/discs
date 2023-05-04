"""Gibbs Sampler Class."""

from itertools import product
import pdb
from discs.common import math_utils as math
from discs.samplers import abstractsampler
import jax
from jax import random
import jax.numpy as jnp
import ml_collections


class BlockGibbsSampler(abstractsampler.AbstractSampler):
  """Gibbs Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.block_size = config.sampler.block_size

  def make_init_state(self, rng):
    """Init sampler state."""
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

  def update_sampler_state(self, sampler_state):
    sampler_state = super.sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + self.block_size) % dim
    sampler_state['num_ll_calls'] += 1
    return sampler_state

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
      x_flatten = x.reshape(1, -1)
      y_flatten = jnp.repeat(
          x_flatten, self.num_categories**self.block_size, axis=0
      )
      categories_iter = jnp.array(
          list(product(range(self.num_categories), repeat=self.block_size))
      )
      y_flatten = y_flatten.at[:, indices_to_flip].set(categories_iter)
      y = y_flatten.reshape((y_flatten.shape[0],) + self.sample_shape)
      return y

    def select_new_samples(model_param, x, y, rnd_categorical):
      loglikelihood = model.forward(model_param, y)
      selected_index = random.categorical(
          rnd_categorical, loglikelihood, axis=1
      )
      x = jnp.take(y, selected_index, axis=0)
      return x

    def loop_body(i, val):
      rng_key, x, indices_to_flip, model_param, cur_samples = val
      curr_sample = x[i]
      y = generate_new_samples(indices_to_flip, curr_sample)
      rnd_categorical, next_key = jax.random.split(rng_key)
      selected_sample = select_new_samples(
          model_param, curr_sample, y, rnd_categorical
      )
      all_samples = jnp.concatenate((cur_samples, selected_sample), axis=0)
      return (next_key, x, indices_to_flip, model_param, all_samples)

    start_index = state['index']
    indices_to_flip = jnp.arange(self.block_size) + start_index
    init_val = (rnd, x, indices_to_flip, model_param, [])
    _, _, _, _, new_x = jax.lax.fori_loop(0, x.shape[0], loop_body, init_val)
    new_state = self.update_sampler_state(state)
    return new_x, new_state, 1


class RBMBlockGibbsSampler(abstractsampler.AbstractSampler):
  """BlockGibbs Sampler specialized for rbm."""

  def make_init_state(self, rng):
    del rng
    return jnp.array((1,))

  def step(self, model, rng, x, model_param, state):
    h_rng, v_rng = jax.random.split(rng)
    h = model.step_h(model_param, h_rng, x)
    new_x = model.step_v(model_param, v_rng, h)
    return new_x, state


def build_sampler(config):
  return BlockGibbsSampler(config)
