"""Gibbs with Grad Sampler Class."""

from discs.samplers import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections
import jax
import pdb
import numpy as np


class GibbsWithGradSampler(abstractsampler.AbstractSampler):
  """Gibbs With Grad Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.adaptive = jnp.where(
        self.num_categories != 2, False, config.sampler.adaptive
    )
    self.target_acceptance_rate = config.sampler.target_acceptance_rate

  def make_init_state(self, rnd):
    """Returns expected number of flips(hamming distance)."""
    num_log_like_calls = 0
    return jnp.array([1.0, num_log_like_calls])

  def step(self, model, rnd, x, model_param, state):
    """Given the current sample, returns the next sample of the chain.

    Args:
      model: target distribution.
      rnd: random key generator for JAX.
      x: current sample.
      model_param: target distribution parameters used for loglikelihood
        calulation.
      state: the state of the sampler.

    Returns:
      New sample.
    """

    def compute_loglike_delta(x, model, model_param):
      _, grad = model.get_value_and_grad(model_param, x)
      if self.num_categories == 2:
        # shape of sample
        return (-2 * x + 1) * grad
      else:
        # shape of sample with categories
        loglike_delta = (jnp.ones(x.shape) - x) * grad
        # hamming ball distance of one
        loglike_delta = loglike_delta - 1e9 * x
        return loglike_delta

    def gumbel_noise(rnd, rate):
      uniform_sample = jax.random.uniform(
          rnd, shape=rate.shape, minval=0, maxval=1
      )
      return -jnp.log(-jnp.log(uniform_sample))

    def categorical_without_replacement(rnd, rate, radius):
      rate = rate + gumbel_noise(rnd, rate)
      idx_flip = jnp.argsort(-rate)
      mask = jnp.full(idx_flip.shape, idx_flip.shape[-1])
      indices = jnp.vstack([jnp.arange(idx_flip.shape[-1])] * rate.shape[0])
      radius = jnp.round(radius)
      idx_flip = jnp.where(indices < radius, idx_flip, mask).astype(jnp.int32)
      return idx_flip

    def sample_index(rnd, loglikelihood, radius):
      loglikelihood_flatten = loglikelihood.reshape(loglikelihood.shape[0], -1)
      if self.adaptive:
        return categorical_without_replacement(
            rnd, loglikelihood_flatten, radius
        )
      else:
        return jnp.expand_dims(
            random.categorical(rnd, loglikelihood_flatten, axis=1), -1
        )

    def update_state(accept_ratio, expected_flips, x):
      clipped_accept_ratio = jnp.clip(accept_ratio, 0.0, 1.0)
      return jnp.minimum(
          jnp.maximum(
              1,
              expected_flips
              + (jnp.mean(clipped_accept_ratio) - self.target_acceptance_rate),
          ),
          x.shape[-1],
      )

    def generate_new_samples(rnd, x, model, model_param, state):
      """Generate the new samples, given the current samples based on the given expected hamming distance.

      Args:
        rnd: key for distribution over index.
        x: current sample.
        model: target distribution.
        model_param: target distribution parameters.

      Returns:
        New samples.
      """
      loglike_delta_x = compute_loglike_delta(x, model, model_param) / 2
      radius = state[0]
      sampled_index_flatten_x = sample_index(rnd, loglike_delta_x, radius)
      # in binary case is the same
      sampled_index_flatten_y = sampled_index_flatten_x

      if self.num_categories == 2:
        flipped = jnp.zeros(x.shape)
        flipped_flatten = flipped.reshape(x.shape[0], -1)
        indx = jnp.expand_dims(jnp.arange(x.shape[0]), -1)
        flipped_flatten = flipped_flatten.at[indx, sampled_index_flatten_x].set(
            jnp.ones(sampled_index_flatten_x.shape)
        )
        flipped = flipped_flatten.reshape(x.shape)
        y = (x + flipped) % self.num_categories
      else:
        # generating new one-hot samples
        sampled_category = sampled_index_flatten_x % self.num_categories
        new_x = jnp.zeros([x.shape[0], self.num_categories])
        new_x = new_x.at[jnp.arange(x.shape[0]), sampled_category].set(
            jnp.ones(x.shape[0])
        )

        dim = np.prod(self.sample_shape)
        y_flatten = x.reshape(x.shape[0], dim, self.num_categories)
        sampled_index = jnp.floor_divide(
            sampled_index_flatten_x, self.num_categories
        )
        # getting sampled i backward
        selected_sample_i = y_flatten[jnp.arange(x.shape[0]), sampled_index]
        index_catogories = jnp.tile(
            jnp.arange(self.num_categories), reps=(x.shape[0], 1)
        )
        sampled_category_back = jnp.sum(
            selected_sample_i * index_catogories, axis=-1
        )
        sampled_index_flatten_y = (
            sampled_index * self.num_categories + sampled_category_back
        ).astype(dtype=jnp.int32)

        # creating the new sample
        y_flatten = y_flatten.at[jnp.arange(x.shape[0]), sampled_index].set(
            new_x
        )
        y = y_flatten.reshape(x.shape)

      return y, sampled_index_flatten_x, sampled_index_flatten_y

    def select_new_samples(
        rnd_acceptance,
        model,
        model_param,
        x,
        y,
        i_flatten_x,
        i_flatten_y,
        state,
    ):
      expected_flips = state[0]
      accept_ratio, state = get_ratio(
          model, model_param, x, y, i_flatten_x, i_flatten_y, state
      )
      accepted = is_accepted(rnd_acceptance, accept_ratio)
      if self.num_categories == 2:
        accepted = accepted.reshape(
            accepted.shape + tuple([1] * len(self.sample_shape))
        )
      else:
        accepted = accepted.reshape(
            accepted.shape + tuple([1] * (len(self.sample_shape) + 1))
        )
      new_x = accepted * y + (1 - accepted) * x
      expected_flips = jnp.where(
          self.adaptive,
          update_state(accept_ratio, expected_flips, x),
          expected_flips,
      )
      state = state.at[0].set(expected_flips)
      return new_x, state

    def logsumexp(values):
      return jnp.log(jnp.sum(jnp.exp(values), -1, keepdims=True))

    def compute_log_probab_index(loglikelihood, i_flatten):
      loglikelihood = loglikelihood.reshape(loglikelihood.shape[0], -1)
      log_probab_index = loglikelihood - logsumexp(loglikelihood)
      indx = jnp.expand_dims(jnp.arange(loglikelihood.shape[0]), -1)
      mask = jnp.full(i_flatten.shape, i_flatten.shape[-1])
      mask_flip = i_flatten != mask
      log_probab = jnp.sum(mask_flip * log_probab_index[indx, i_flatten], -1)
      return log_probab

    def get_ratio(model, model_param, x, y, i_flatten_x, i_flatten_y, state):
      loglike_delta_x = compute_loglike_delta(x, model, model_param) / 2
      loglike_delta_y = compute_loglike_delta(y, model, model_param) / 2

      probab_i_given_x = compute_log_probab_index(loglike_delta_x, i_flatten_x)
      probab_i_given_y = compute_log_probab_index(loglike_delta_y, i_flatten_y)

      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)
      state = state.at[1].set(state[1] + 2)

      return (
          jnp.exp(
              loglikelihood_y
              - loglikelihood_x
              + probab_i_given_y
              - probab_i_given_x
          ),
          state,
      )

    def is_accepted(rnd_acceptance, accept_ratio):
      random_uniform_val = random.uniform(
          rnd_acceptance, shape=accept_ratio.shape, minval=0.0, maxval=1.0
      )
      return jnp.where(accept_ratio >= random_uniform_val, 1, 0)

    if self.num_categories != 2:
      x = jax.nn.one_hot(x, self.num_categories)
    rnd_new_sample, rnd_acceptance = random.split(rnd)
    del rnd
    y, i_flatten_x, i_flatten_y = generate_new_samples(
        rnd_new_sample, x, model, model_param, state
    )
    new_x, new_state = select_new_samples(
        rnd_acceptance,
        model,
        model_param,
        x,
        y,
        i_flatten_x,
        i_flatten_y,
        state,
    )

    if self.num_categories != 2:
      new_x = jnp.argmax(new_x, axis=-1)

    return new_x, new_state


def build_sampler(config):
  return GibbsWithGradSampler(config)
