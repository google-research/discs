"""Hamming Ball Sampler Class."""

from discs.common import math_util as math
from discs.samplers import abstractsampler
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import pdb

class HammingBallSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Base Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.num_categories = config.model.num_categories
    self.hamming = config.sampler.hamming
    self.block_size = config.sampler.block_size

  def make_init_state(self, rng):
    state = super().make_init_state(rng)
    state['radius'] = jnp.ones(shape=(), dtype=jnp.float32)
    pdb.set_trace()
    if self.num_categories == 2:
      num_samples_per_hamming = [
          math.comb(self.block_size, j + 1) for j in range(self.hamming)
      ]
      num_samples_per_hamming = jnp.array([1] + num_samples_per_hamming)
      self.r_init = jax.random.categorical(rng, num_samples_per_hamming)
    else:
      num_samples_per_hamming = [
          math.comb(self.block_size, j + 1)
          * (self.num_categories - 1) ** (j + 1)
          for j in range(self.hamming)
      ]
      num_samples_per_hamming = jnp.array([1] + num_samples_per_hamming)
      self.r_init = jax.random.categorical(rng, num_samples_per_hamming)
   
    return state

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    return new_x, new_state, acc


def build_sampler(config):
  return HammingBallSampler(config)
