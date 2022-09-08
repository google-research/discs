"""Random Walk Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections


class RandomWalkSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.adaptive = config.adaptive
    self.target_acceptance_rate = config.target_acceptance_rate
    self.sample_dimension = config.sample_dimension
    self.num_categories = config.num_categories

  def make_init_state(self, rnd):
    """Returns expected number of flips."""
    return 1  #random.uniform(rnd, shape=(1, 1), minval=1, maxval=self.sample_dimension).at[0, 0].get()

  def step(self, model, rnd, x, model_param, state):
    """Given the current sample, returns the next sample of the chain.

    Args:
      model: target distribution.
      rnd: random key generator for JAX.
      x: current sample.
      model_param: target distribution parameters used for loglikelihood
        calulation.
      state: the state of the sampler (changes in adaptive case to tune the
        proposal distribution).

    Returns:
      New sample.
    """

    def get_new_sample(rnd_new_sample, x, expected_num_flips):
      """Proposal distribution to sample the next state.

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
              maxval=self.num_categories)
      return (flipped + x) % self.num_categories

    def get_accept_ratio(model, model_param, x, y):
      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)
      return jnp.exp(loglikelihood_y - loglikelihood_x)

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
              (jnp.mean(clipped_accept_ratio) - self.target_acceptance_rate)),
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
