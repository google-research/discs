"""Gibbs sampler."""

from discs.samplers import abstractsampler
import jax
import jax.numpy as jnp
import ml_collections


class GibbsSampler(abstractsampler.AbstractSampler):
  """Gibbs Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories

  def make_init_state(self, rng):
    """Init sampler state."""
    state = super().__init__(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

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

    def get_next_sample(i, val):
      rng_key, cur_ll, cur_sample = val
      cur_ll = jnp.expand_dims(cur_ll, axis=0)
      all_new_scores = get_ll_at_dim(cur_sample, i)
      all_scores = jnp.concatenate((cur_ll, all_new_scores), axis=0)
      cur_key, _ = jax.random.split(rng_key)
      val_change = jax.random.categorical(cur_key, all_scores, axis=0)
      y = cur_sample.at[:, i].add(val_change) % self.num_categories
      return y

    init_val = (rng, init_ll, x)
    index = state['index']
    y = get_next_sample(index, init_val)
    y = jnp.reshape(y, x_shape)
    num_calls = x.shape[-1] * (self.num_categories - 1) + 1
    sampler_state = state
    state['num_ll_calls'] += num_calls
    state['index'] += (state['index'] + 1) % x.shape[-1]

    return y, sampler_state, 1


def build_sampler(config):
  return GibbsSampler(config)
