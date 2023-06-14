"""Random Walk Sampler Class."""

from discs.common import math_util as math
from discs.samplers import abstractsampler
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import pdb

class RandomWalkSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Base Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.adaptive = config.sampler.adaptive
    self.target_acceptance_rate = config.sampler.target_acceptance_rate
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories

  def make_init_state(self, rng):
    """Returns expected number of flips(hamming distance) and the number ."""
    state = super().make_init_state(rng)
    state['radius'] = jnp.ones(shape=(), dtype=jnp.float32)
    return state

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    rng_new_sample, rng_acceptance = jax.random.split(rng)
    ll_x = model.forward(model_param, x)
    y = self.sample_from_proposal(rng_new_sample, x, state)
    ll_y = model.forward(model_param, y)
    log_acc = ll_y - ll_x
    new_x, new_state = self.select_sample(rng_acceptance, log_acc, x, y, state)
    acc = jnp.mean(jnp.clip(jnp.exp(log_acc), a_max=1))
    return new_x, new_state, acc

  def sample_from_proposal(self, rng_new_sample, x, state):
    rnd_new_sample, rnd_new_sample_randint = random.split(rng_new_sample)
    dim = math.prod(self.sample_shape)
    indices_to_flip = random.bernoulli(
        rnd_new_sample, p=(state['radius'] / dim), shape=x.shape
    )
    flipping_value = indices_to_flip * random.randint(
        rnd_new_sample_randint,
        shape=x.shape,
        minval=1,
        maxval=self.num_categories,
    )
    return (flipping_value + x) % self.num_categories


class RWSampler(RandomWalkSampler):

  def select_sample(
      self, rng, log_acc, current_sample, new_sample, sampler_state
  ):
    y, acc = math.mh_step(rng, log_acc, current_sample, new_sample)
    sampler_state['num_ll_calls'] += 2
    sampler_state['acceptance_rate'] = jnp.mean(acc.astype(jnp.float32))
    return y.astype(jnp.int32), sampler_state


class AdaptiveRWSampler(RWSampler):

  def select_sample(
      self, rng, log_acc, current_sample, new_sample, sampler_state
  ):
    y, new_state = super().select_sample(
        rng, log_acc, current_sample, new_sample, sampler_state
    )
    acc = jnp.mean(jnp.exp(jnp.clip(log_acc, a_max=0.0)))
    r = sampler_state['radius'] + acc - self.target_acceptance_rate
    new_state['radius'] = jnp.clip(r, a_min=1, a_max=math.prod(y.shape[1:]))
    return y, new_state


def build_sampler(config):
  if config.sampler.get('adaptive', False):
    return AdaptiveRWSampler(config)
  else:
    return RWSampler(config)
