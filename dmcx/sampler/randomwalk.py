"""Random Walk Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections


class RandomWalkSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.adaptive = config.adaptive
    self.ber_target_accept_rate = config.ber_target_accept_rate
    self.sample_dimension = config.sample_dimension
    self.num_categ = config.num_categ

  def make_init_state(self, rnd):
    """Returns expected number of flips."""
    return 1  #random.uniform(rnd, shape=(1, 1), minval=1, maxval=10).at[0, 0].get()

  def step(self, model, rnd, x, model_param, state):

    def get_new_sample(rnd_new_sample, x, expected_num_flips):
      """Get new sample.

      Args:
        rnd_new_sample: key for binary mask and random flip.
        x: current sample.
        expected_num_flips: expected number of indices to flip.

      Returns:
        New sample.
      """
      rnd_new_sample, rnd_new_sample_randint = random.split(rnd_new_sample)
      flipped = random.bernoulli(
          rnd_new_sample, p=(expected_num_flips / x.shape[-1]),
          shape=x.shape) * random.randint(
              rnd_new_sample_randint,
              shape=x.shape,
              minval=1,
              maxval=self.num_categ)
      return (flipped + x) % self.num_categ

    def get_accept_ratio(model, model_param, x, y):
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
              (jnp.mean(clipped_accept_ratio) - self.ber_target_accept_rate)),
          x.shape[-1])

    rnd_new_sample, rnd_acceptance = random.split(rnd)
    del rnd
    y = get_new_sample(rnd_new_sample, x, state)
    accept_ratio = get_accept_ratio(model, model_param, x, y)
    accepted = is_accepted(rnd_acceptance, accept_ratio)
    new_x = accepted * y + (1 - accepted) * x
    new_state = jnp.where(self.adaptive, update_state(accept_ratio, state, x),
                          state)
    return new_x, new_state
