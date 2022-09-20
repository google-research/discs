"""Random Walk Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections
import math
import pdb


class RandomWalkSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.adaptive = config.adaptive
    self.target_acceptance_rate = config.target_acceptance_rate
    if isinstance(config.sample_shape, int):
      self.sample_shape = (config.sample_shape,)
    else:
      self.sample_shape = config.sample_shape
    self.num_categories = config.num_categories
    self.num_dimension = len(self.sample_shape)

  def make_init_state(self, rnd):
    """Returns expected number of flips(hamming distance)."""
    return 1  #random.uniform(rnd, shape=(1, 1), minval=1, maxval=self.sample_shape).at[0, 0].get()

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

    def generate_new_samples(rnd_new_sample, x, expected_hamming_distance):
      """Generate the new samples, given the current samples based on the given expected hamming distance.

      Args:
        rnd_new_sample: key for binary mask and random flip.
        x: current sample.
        expected_hamming_distance: expected number of indices to flip.

      Returns:
        New samples.
      """
      rnd_new_sample, rnd_new_sample_randint = random.split(rnd_new_sample)
      dim = math.prod(self.sample_shape)
      indices_to_flip = random.bernoulli(
          rnd_new_sample,
          p=(expected_hamming_distance / dim),
          shape=x.shape)
      flipping_value = indices_to_flip * random.randint(
          rnd_new_sample_randint,
          shape=x.shape,
          minval=1,
          maxval=self.num_categories)
      return (flipping_value + x) % self.num_categories

    def select_new_samples(model, model_param, x, y, state):
      accept_ratio = get_ratio(model, model_param, x, y)
      accepted = is_accepted(rnd_acceptance, accept_ratio)
      accepted = accepted.reshape(accepted.shape +
                                  tuple([1] * len(self.sample_shape)))
      new_x = accepted * y + (1 - accepted) * x
      new_state = jnp.where(self.adaptive, update_state(accept_ratio, state, x),
                            state)
      return new_x, new_state

    def get_ratio(model, model_param, x, y):
      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)
      return jnp.exp(loglikelihood_y - loglikelihood_x)

    def is_accepted(rnd_acceptance, accept_ratio):
      random_uniform_val = random.uniform(
          rnd_acceptance, shape=accept_ratio.shape, minval=0.0, maxval=1.0)
      return jnp.where(accept_ratio >= random_uniform_val, 1, 0)

    def update_state(accept_ratio, expected_flips, x):
      clipped_accept_ratio = jnp.clip(accept_ratio, 0.0, 1.0)
      return jnp.minimum(
          jnp.maximum(
              1, expected_flips +
              (jnp.mean(clipped_accept_ratio) - self.target_acceptance_rate)),
          x.shape[-1])

    rnd_new_sample, rnd_acceptance = random.split(rnd)
    del rnd
    y = generate_new_samples(rnd_new_sample, x, state)
    assert y.shape == x.shape
    new_x, new_state = select_new_samples(model, model_param, x, y, state)
    assert new_x.shape == x.shape
    
    return new_x, new_state
