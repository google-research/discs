"""Random Walk Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections
import jax
from functools import partial


class RandomWalkSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.adaptive = config.adaptive
    self.target_accept_ratio = config.target_accept_ratio
    self.sample_dimension = config.sample_dimension

  def make_init_state(self, rnd):
    return random.randint(
        rnd,
        shape=(1, 1),
        minval=0,
        maxval=self.sample_dimension,
        dtype=jnp.int32).at[0, 0].get()

  def step(self, model, rnd, x, model_param, state):

    def get_num_flips(rnd_uniform, expected_flips):
      return jnp.where(
          random.uniform(rnd_uniform, minval=0.0, maxval=1.0) <
          expected_flips - jnp.floor(expected_flips),
          jnp.floor(expected_flips).astype(int),
          jnp.ceil(expected_flips).astype(int))

    def get_new_sample(rnd_new_sample, x, num_flips):
      """Get new sample.

      Args:
        rnd_new_sample: key for permutation.
        x: current sample.
        num_flips: number of indices to flip.

      Returns:
        New sample.
      """
      flipped = random.bernoulli(
          rnd_new_sample,
          p=(num_flips / x.shape[-1]),
          shape=[x.shape[0], x.shape[-1]])
      return jnp.where(flipped, 1 - x, x)

    def get_accept_ratio(model_param, x, y):
      e_x = model.forward(model_param, x)
      e_y = model.forward(model_param, y)
      return jnp.exp(-e_y + e_x)

    def is_accepted(rnd_acceptance, accept_ratio):
      random_uniform_val = random.uniform(
          rnd_acceptance, shape=accept_ratio.shape, minval=0.0, maxval=1.0)
      return jnp.expand_dims(
          jnp.where(accept_ratio >= random_uniform_val, 1, 0), axis=-1)

    def update_state(accept_ratio, expected_flips, x):
      clipped_accept_ratio = jnp.clip(accept_ratio, 0.0, 1.0)
      return jnp.minimum(
          jnp.maximum(
              1, expected_flips +
              (jnp.mean(clipped_accept_ratio) - self.target_accept_ratio)),
          x.shape[-1])

    rnd_uniform, rnd_new_sample, rnd_acceptance = random.split(rnd, num=3)
    del rnd
    num_flips = jnp.where(self.adaptive, get_num_flips(rnd_uniform, state),
                          jnp.floor(state).astype(int))
    y = get_new_sample(rnd_new_sample, x, num_flips)
    accept_ratio = get_accept_ratio(model_param, x, y)
    accepted = is_accepted(rnd_acceptance, accept_ratio)
    new_x = accepted * y + (1 - accepted) * x
    new_state = jnp.where(self.adaptive, update_state(accept_ratio, state, x),
                          state)
    return new_x, new_state
