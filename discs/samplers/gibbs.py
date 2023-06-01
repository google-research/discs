"""Gibbs sampler."""

from discs.common import math_util as math
from discs.samplers import abstractsampler
import jax
import jax.numpy as jnp
import ml_collections
import pdb


class GibbsSampler(abstractsampler.AbstractSampler):
  """Gibbs Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    
  def make_init_state(self, rng):
    """Init sampler state."""
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

  def update_sampler_state(self, sampler_state):
    sampler_state = super().update_sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + 1) % dim
    sampler_state['num_ll_calls'] += self.num_categories
    return sampler_state

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    x_shape = x.shape
    init_ll = model.forward(model_param, x)
    x = jnp.reshape(x, (x.shape[0], -1))

    def get_ll_at_dim(cur_sample, dim):
      def fn_ll(_, i):
        x_new = cur_sample.at[:, dim].add(i) % self.num_categories
        ll = model.forward(model_param, jnp.reshape(x_new, x_shape))
        return None, ll

      _, ll_all = jax.lax.scan(fn_ll, None, jnp.arange(1, self.num_categories))
      return ll_all

    def compute_next_x(rng_key, cur_ll, cur_sample, index):
      cur_ll = jnp.expand_dims(cur_ll, axis=0)
      all_new_scores = get_ll_at_dim(cur_sample, index)
      all_scores = jnp.concatenate((cur_ll, all_new_scores), axis=0)
      val_change = jax.random.categorical(rng_key, all_scores, axis=0)
      y = cur_sample.at[:, index].add(val_change) % self.num_categories
      y = jnp.reshape(y, (-1,)+self.sample_shape)
      return y

    index = state['index']
    new_x = compute_next_x(rng, init_ll, x, index)
    new_state = self.update_sampler_state(state)
    return new_x, new_state, 1


def build_sampler(config):
  return GibbsSampler(config)
