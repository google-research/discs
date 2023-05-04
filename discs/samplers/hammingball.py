"""Hamming Ball Sampler Class."""

from discs.common import math_util as math
from discs.samplers import abstractsampler
import jax
from jax import random
import jax.numpy as jnp
import ml_collections

class HammingBallSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Base Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.num_categories = config.model.num_categories

  def make_init_state(self, rng):
    state = super().make_init_state(rng)
    state['radius'] = jnp.ones(shape=(), dtype=jnp.float32)
    return state

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    return new_x, new_state, acc
 

def build_sampler(config):
  return HammingBallSampler(config)
