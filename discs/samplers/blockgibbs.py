"""Block gibbs Sampler Class."""

from itertools import product
from discs.common import math_util as math
from discs.samplers import abstractsampler
import jax
from jax import random
import jax.numpy as jnp
import ml_collections


class BlockGibbsSampler(abstractsampler.AbstractSampler):
  """Block gibbs Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.block_size = config.sampler.block_size
    if self.block_size != 1:
      self.categories_iter = jnp.array(
          list(product(range(self.num_categories), repeat=self.block_size))
      )
    else:
      self.categories_iter = jnp.arange(self.num_categories).reshape([-1,1])

  def make_init_state(self, rng):
    """Init sampler state."""
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

  def update_sampler_state(self, sampler_state):
    sampler_state = super().update_sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + self.block_size) % dim
    sampler_state['num_ll_calls'] += self.num_categories**self.block_size
    return sampler_state

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask

    def get_ll_at_block(indices_to_flip, x, model_param):
      def fn_ll(_, i):
        y_flatten = x.reshape(x.shape[0], -1)
        y_flatten = y_flatten.at[:, indices_to_flip].set(
            self.categories_iter[i]
        )
        y = y_flatten.reshape((-1,)+self.sample_shape)
        ll = model.forward(model_param, y)
        return None, ll

      _, ll_all = jax.lax.scan(
          fn_ll, None, jnp.arange(0, len(self.categories_iter))
      )
      return ll_all

    def select_new_samples(rnd_categorical, loglikelihood, x, indices_to_flip):
      selected_index = random.categorical(rnd_categorical, loglikelihood, axis=0)
      vals = jnp.take(self.categories_iter, selected_index, axis=0)
      x_flatten = x.reshape(x.shape[0], -1)
      x = x_flatten.at[:, indices_to_flip].set(vals)
      x = x.reshape((-1,)+self.sample_shape)
      return x

    start_index = state['index']
    indices_to_flip = jnp.arange(self.block_size) + start_index
    loglikelihood = get_ll_at_block(indices_to_flip, x, model_param)
    new_x = select_new_samples(rng, loglikelihood, x, indices_to_flip)

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
