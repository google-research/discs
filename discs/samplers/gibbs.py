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

  def step(self, model, rng, x, model_param, state):
    """Given the current sample, returns the next sample of the chain.

    Args:
      model: target distribution.
      rng: random key generator for JAX.
      x: current sample.
      model_param: target distribution parameters used for loglikelihood
        calulation.
      state: the state of the sampler.

    Returns:
      New sample.
    """
    x_shape = x.shape
    x = jnp.reshape(x, (x.shape[0], -1))
    cur_ll = jnp.expand_dims(
        model.forward(model_param, jnp.reshape(x, x_shape)), axis=0)

    def get_ll_at_dim(dim):
      def fn_ll(_, i):
        x_new = x.at[:, dim].add(i) % self.num_categories
        ll = model.forward(model_param, jnp.reshape(x_new, x_shape))
        return None, ll
      _, ll_all = jax.lax.scan(fn_ll, None, jnp.arange(1, self.num_categories))
      return ll_all

    def loop_body(i, val):
      rng_key, cur_sample = val
      all_new_scores = get_ll_at_dim(i)
      all_scores = jnp.concatenate((cur_ll, all_new_scores), axis=0)
      cur_key, next_key = jax.random.split(rng_key)
      val_change = jax.random.categorical(cur_key, all_scores, axis=0)
      y = cur_sample.at[:, i].add(val_change) % self.num_categories
      return (next_key, y)

    init_val = (rng, x)
    _, y = jax.lax.fori_loop(0, x.shape[-1], loop_body, init_val)
    y = jnp.reshape(y, x_shape)
    num_calls = x.shape[-1] * (self.num_categories - 1) + 1
    sampler_state = {
        'num_ll_calls': state['num_ll_calls'] + num_calls,
    }
    return y, sampler_state


def build_sampler(config):
  return GibbsSampler(config)
