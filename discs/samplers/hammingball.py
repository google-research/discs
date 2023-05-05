"""Hamming Ball Sampler Class."""

import copy
import pdb
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
    self.hamming = config.sampler.hamming
    self.block_size = config.sampler.block_size
    assert self.hamming <= self.block_size
    self.hamming_logit = jnp.ones(1)
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
    self.hamming_logit += jnp.array(num_samples_per_hamming)

    self.choose_index_vmapped = jax.vmap(
        self.choose_index, in_axes=[0, None, None]
    )

  def choose_index(self, rng, arr, rad):
    res = jax.random.choice(rng, arr, shape=(rad,), replace=False)
    return res

  def update_sampler_state(self, sampler_state):
    sampler_state = super().update_sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + self.block_size) % dim
    # sampler_state['num_ll_calls'] += (self.num_categories**self.block_size)
    return sampler_state

  def make_init_state(self, rng):
    """Init sampler state."""
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

  def compute_u(self, rng, rad, x):
    rng_ber, rng_int = random.split(rng)
    indices_to_flip = random.bernoulli(
        rng_ber, p=(rad / self.block_size), shape=[x.shape[0], self.block_size]
    )
    flipping_value = indices_to_flip * random.randint(
        rng_int,
        shape=x.shape,
        minval=1,
        maxval=self.num_categories,
    )
    u = (flipping_value + x) % self.num_categories
    return u
      
  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    rad = jax.random.categorical(rng, self.hamming_logit)
    start_index = state['index']
    block = start_index + jnp.arange(self.block_size)
    u = jnp.where(rad, self.compute_u(rng, rad, x), x)

    # Y = u.unsqueeze(1)
    # for j in range(self.hamming):
    #     y = torch.stack([u.clone() for _ in range(comb(self.block_size, j+1))], dim=1)
    #     idx_pos = torch.combinations(block, j + 1)
    #     idx_p = torch.arange(idx_pos.shape[0]).unsqueeze(-1)
    #     y[:, idx_p, idx_pos] = 1 - y[:, idx_p, idx_pos]
    #     Y = torch.cat([Y, y], dim=1)

    # energy = model(Y)
    # selected = torch.multinomial(torch.softmax(energy, dim=-1), 1)
    # new_x = Y[b_idx, selected].squeeze()
    # hop = torch.abs(new_x - x).mean(0).sum().item()
    # self.update_stats(length=2 * self.hamming, acc=1, hop=hop)
    # return new_x

    new_state = self.update_sampler_state(state)
    return new_x, new_state, acc


def build_sampler(config):
  return HammingBallSampler(config)
