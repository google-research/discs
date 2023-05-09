"""Hamming Ball Sampler Class."""

import copy
from itertools import product
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
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.hamming = 1  # config.sampler.hamming
    self.block_size = config.sampler.block_size
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

  def update_sampler_state(self, sampler_state):
    sampler_state = super().update_sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + self.block_size) % dim
    sampler_state['num_ll_calls'] += self.num_categories**self.hamming
    return sampler_state

  def make_init_state(self, rng):
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

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
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    state_init = copy.deepcopy(state)
    _ = x_mask
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    rad = jax.random.categorical(rng1, self.hamming_logit)
    start_index = state['index']
    block = start_index + jnp.arange(self.block_size)
    u = jnp.where(rad, self.compute_u(rng2, rad, x, block), x)
    u = jnp.reshape(u, x_shape)

    def generate_new_samples(indices_to_flip, x):
      x_flatten = x.reshape(1, -1)
      y_flatten = jnp.repeat(
          x_flatten, (self.num_categories - 1) * self.block_size, axis=0
      )
      indices_to_flip = jnp.repeat(
          indices_to_flip, self.num_categories - 1, axis=0
      )
      categories_iter = jnp.tile(
          jnp.arange(1, self.num_categories), self.block_size
      )
      b_idx = jnp.arange(y_flatten.shape[0])
      y_flatten = y_flatten.at[b_idx, indices_to_flip].set(
          (y_flatten[b_idx, indices_to_flip] + categories_iter)
          % self.num_categories
      )
      y = y_flatten.reshape((y_flatten.shape[0],) + self.sample_shape)
      return y

    def select_new_samples(model_param, x, y, rnd_categorical):
      loglikelihood = model.forward(model_param, y)
      selected_index = random.categorical(rnd_categorical, loglikelihood)
      x = jnp.take(y, selected_index, axis=0)
      x = x.reshape(self.sample_shape)
      return x

    def loop_body(i, val):
      rng_key, x, indices_to_flip, model_param = val
      curr_sample = x[i]
      y = generate_new_samples(indices_to_flip, curr_sample)
      y_all = jnp.concatenate([jnp.array([curr_sample]), y], axis=0)
      rnd_categorical, next_key = jax.random.split(rng_key)
      selected_sample = select_new_samples(
          model_param, curr_sample, y_all, rnd_categorical
      )
      x = x.at[i].set(selected_sample)
      return (next_key, x, indices_to_flip, model_param)

    init_val = (rng3, u, block, model_param)
    _, new_x, _, _ = jax.lax.fori_loop(0, x.shape[0], loop_body, init_val)
    new_state = self.update_sampler_state(state_init)
    return new_x, new_state, 1


def build_sampler(config):
  return HammingBallSampler(config)
