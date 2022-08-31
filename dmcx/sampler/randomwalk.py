"""Random Walk Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections


class RandomWalkSampler(abstractsampler.AbstractSampler):
  """Random Walk Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.adaptive = config.adaptive
    self.target_accept_ratio = config.target_accept_ratio

  def step(self, rnd, x, model, model_param, state):

    def get_num_flips(rnd_uniform, expected_flips):
      if random.uniform(
          rnd_uniform, minval=0.0,
          maxval=1.0) < expected_flips - jnp.floor(expected_flips):
        num_flips = int(jnp.floor(expected_flips))
      else:
        num_flips = int(jnp.ceil(expected_flips))
      return num_flips

    def get_new_sample(rnd_new_sample, x, num_flips):
      """Get new sample.

      Args:
        rnd_new_sample: key for permutation.
        x: current sample.
        num_flips: number of indices to flip.

      Returns:
        New sample.
      """

      index_axis0 = jnp.repeat(jnp.arange(x.shape[0]), num_flips)
      aranged_index_batch = jnp.tile(jnp.arange(x.shape[-1]), [x.shape[0], 1])
      index_axis1 = random.permutation(
          rnd_new_sample, aranged_index_batch, axis=-1,
          independent=True)[:, 0:num_flips].reshape(1, -1)
      zeros = jnp.zeros(x.shape)
      flipped = zeros.at[index_axis0, index_axis1].set(1)
      return jnp.where(x + flipped == 2, 0, x + flipped)

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
      return min(
          max([
              1, expected_flips +
              (jnp.mean(clipped_accept_ratio) - self.target_accept_ratio)
          ]), x.shape[-1])

    rnd_uniform, rnd_new_sample, rnd_acceptance = random.split(rnd, num=3)
    del rnd
    num_flips = get_num_flips(rnd_uniform, state)

    if x.shape[-1] < num_flips:
      raise Exception(
          "Determined state should be smaller than dimension of the samples."
      )

    y = get_new_sample(rnd_new_sample, x, num_flips)
    accept_ratio = get_accept_ratio(model_param, x, y)
    accepted = is_accepted(rnd_acceptance, accept_ratio)
    new_x = accepted * y + (1 - accepted) * x
    new_state = state
    if self.adaptive:
      new_state = update_state(accept_ratio, state, x)

    return new_x, new_state
