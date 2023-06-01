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
    self.block_size = config.sampler.block_size
    self.hamming = 1
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
    self.choose_index_vmapped = jax.vmap(self.choose_index, in_axes=[0, None])

  def choose_index(self, rng, arr):
    res = jax.random.choice(rng, arr, shape=(1,), replace=False)
    return res

  def update_sampler_state(self, sampler_state):
    sampler_state = super().update_sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + self.block_size) % dim
    sampler_state['num_ll_calls'] += (self.num_categories - 1) * (
        self.block_size
    ) + 1
    return sampler_state

  def make_init_state(self, rng):
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

  def compute_u(self, rng, x, block):
    rng_spl, rng_int = random.split(rng)
    flipping_value = random.randint(
        rng_int,
        shape=[x.shape[0]],
        minval=1,
        maxval=self.num_categories,
    )
    rng_v = jax.random.split(rng_spl, x.shape[0])
    flip_index = self.choose_index_vmapped(rng_v, block)
    flip_index = jnp.reshape(flip_index, [-1])
    b_idx = jnp.arange(x.shape[0])
    u = x.at[b_idx, flip_index].set(
        (x[b_idx, flip_index] + flipping_value) % self.num_categories
    )
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
    u = jnp.where(rad, self.compute_u(rng2, x, block), x)
    u = jnp.reshape(u, x_shape)

    def get_ll_at_block(indices_to_flip, x, model_param):
      def fn_ll(_, i):
        y_flatten = x.reshape(x.shape[0], -1)
        index = i // self.block_size
        categ = 1 + i % (self.num_categories - 1)
        val = (
            y_flatten[:, indices_to_flip[index]] + categ
        ) % self.num_categories
        y_flatten = y_flatten.at[:, indices_to_flip[index]].set(val)
        y = y_flatten.reshape((-1,) + self.sample_shape)
        ll = model.forward(model_param, y)
        return None, ll

      num_neighs = (self.num_categories - 1) * self.block_size
      _, ll_all = jax.lax.scan(fn_ll, None, jnp.arange(num_neighs))
      return ll_all

    def select_new_samples(rnd_categorical, loglikelihood, x, indices_to_flip):
      selected_index = random.categorical(
          rnd_categorical, loglikelihood, axis=0
      )
      index = selected_index // self.block_size
      categ = 1 + selected_index % (self.num_categories - 1)
      x_flatten = x.reshape(x.shape[0], -1)
      val = (x_flatten[:, indices_to_flip[index]] + categ) % self.num_categories
      x = x_flatten.at[:, indices_to_flip[index]].set(val)
      x = x.reshape((-1,) + self.sample_shape)
      return x

    loglikelihood = get_ll_at_block(block, u, model_param)
    new_x = select_new_samples(rng3, loglikelihood, u, block)
    new_state = self.update_sampler_state(state_init)
    return new_x, new_state, 1


def build_sampler(config):
  return HammingBallSampler(config)
