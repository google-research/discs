"""Hamming Ball Sampler Class."""

import copy
import pdb
from discs.common import math_util as math
from discs.samplers import abstractsampler
from discs.samplers import blockgibbs
import jax
from jax import random
import jax.numpy as jnp
import ml_collections


class HammingBallSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Base Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.num_categories = config.model.num_categories
    self.hamming = config.sampler.hamming
    self.block_size = config.sampler.block_size
    config_bg = copy.deepcopy(config)
    config_bg.sampler.block_size = self.hamming
    self.blockgibbs = blockgibbs.build_sampler(config_bg)
    assert self.hamming <= self.block_size
    self.hamming_logit = [1.0]
    if self.num_categories == 2:
      num_samples_per_hamming = [
          math.comb(self.block_size, j + 1) for j in range(self.hamming)
      ]
    else:
      num_samples_per_hamming = [
          math.comb(self.block_size, j + 1)
          * (self.num_categories - 1) ** (j + 1)
          for j in range(self.hamming)
      ]
    self.hamming_logit = jnp.array(self.hamming_logit + num_samples_per_hamming)


  self.choose_index_vmapped = jax.vmap(
        self.choose_index, in_axes=[0, None, None]
    )

  def choose_index(self, rng, arr, rad):
    res = jax.random.choice(rng, arr, shape=(rad,), replace=False)
    return res
  
  def update_sampler_state(self, sampler_state):
    return self.blockgibbs.update_sampler_state(sampler_state)

  def make_init_state(self, rng):
    return self.blockgibbs.make_init_state(rng)

  def compute_u(self, rng, rad, x, block):
    rng_ber, rng_int = random.split(rng)
    indices_to_flip = random.bernoulli(
        rng_ber, p=(rad / self.block_size), shape=[x.shape[0], self.block_size]
    )
    flipping_value = indices_to_flip * random.randint(
        rng_int,
        shape=[x.shape[0], self.block_size],
        minval=1,
        maxval=self.num_categories,
    )
    u = x.at[:, block].set((x[:, block] + flipping_value) % self.num_categories)
    return u

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    x = x.reshape(x.shape[0], -1)
    b_idx = jnp.arange(x.shape[0])
    rad = jax.random.categorical(rng, self.hamming_logit)
    start_index = state['index']
    block = start_index + jnp.arange(self.block_size)
    rng_v = jax.random.split(rng, x.shape[0])
    indices_flip = self.choose_index_vmapped(rng_v, block, rad)
    indices = (jnp.zero_like(block)-1)
    indices += indices_flip
    u[b_idx, block[indices]] = 1 - u[b_idx, block[indices]]
    
  
    
    
    u = jnp.where(rad, self.compute_u(rng, rad, x, block), x)
    return self.blockgibbs.step(model, rng, u, model_param, state)


def build_sampler(config):
  return HammingBallSampler(config)
