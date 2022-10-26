"""Gibbs with Grad Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections
import math
import pdb


class GibbsWithGradSampler(abstractsampler.AbstractSampler):
  """Gibbs With Grad Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):

    self.sample_shape = config.sample_shape
    self.num_categories = config.num_categories

  def make_init_state(self, rnd):
    """Returns expected number of flips(hamming distance)."""
    return 1

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

    def sample_index(rnd_categorical, loglikelihood):
      loglikelihood_flatten = loglikelihood.reshape(loglikelihood.shape[0], -1)
      return random.categorical(rnd_categorical, loglikelihood_flatten, axis=1)

    def generate_new_samples(rnd, x, model, model_param):
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
      sampled_index_flatten_x = sample_index(rnd, loglike_delta_x)
      sampled_index_flatten_y = sampled_index_flatten_x

      if self.num_categories == 2:
        flipped = jnp.zeros(x.shape)
        flipped_flatten = flipped.reshape(x.shape[0], -1)
        flipped_flatten = flipped_flatten.at[jnp.arange(x.shape[0]),
                                             sampled_index_flatten_x].set(
                                                 jnp.ones(x.shape[0]))
        flipped = flipped_flatten.reshape(x.shape)
        y = (x + flipped) % self.num_categories
      else:
        dim = math.prod(self.sample_shape)
        sampled_index_from_shape = jnp.floor_divide(sampled_index_flatten_x,
                                                    self.num_categories)
        sampled_category = sampled_index_flatten_x % self.num_categories
        # generating new one hots
        new_x = jnp.zeros([x.shape[0], self.num_categories])
        new_x = new_x.at[jnp.arange(x.shape[0]),
                         sampled_category].set(jnp.ones(x.shape[0]))
        y_flatten = x.reshape(x.shape[0], dim, self.num_categories)
        # getting sampled i backward
        selected_i = y_flatten[jnp.arange(x.shape[0]), sampled_index_from_shape]
        index_catogories = jnp.tile(
            jnp.arange(self.num_categories), reps=(x.shape[0], 1))
        sampled_category_back = jnp.sum(selected_i * index_catogories, axis=-1)
        sampled_index_flatten_y = sampled_index_from_shape * self.num_categories + sampled_category_back

        # creating the new sample
        y_flatten = y_flatten.at[jnp.arange(x.shape[0]),
                                 sampled_index_from_shape].set(new_x)
        y = y_flatten.reshape(x.shape)

      return y, sampled_index_flatten_x, sampled_index_flatten_y

    def select_new_samples(rnd_acceptance, model, model_param, x, y,
                           i_flatten_x, i_flatten_y):

      accept_ratio = get_ratio(model, model_param, x, y, i_flatten_x,
                               i_flatten_y)
      accepted = is_accepted(rnd_acceptance, accept_ratio)
      if self.num_categories == 2:
        accepted = accepted.reshape(accepted.shape +
                                    tuple([1] * len(self.sample_shape)))
      else:
        accepted = accepted.reshape(accepted.shape +
                                    tuple([1] * (len(self.sample_shape) + 1)))
      new_x = accepted * y + (1 - accepted) * x
      return new_x

    def compute_softmax(loglikelihood):
      return jnp.exp(loglikelihood) / jnp.sum(
          jnp.exp(loglikelihood), axis=1, keepdims=True)

    def compute_probab_index(loglikelihood, i_flatten):

      loglikelihood = loglikelihood.reshape(loglikelihood.shape[0], -1)
      probability = compute_softmax(loglikelihood)
      return probability[jnp.arange(loglikelihood.shape[0]), i_flatten]

    def get_ratio(model, model_param, x, y, i_flatten_x, i_flatten_y):

      loglike_delta_x = compute_loglike_delta(x, model, model_param) / 2
      loglike_delta_y = compute_loglike_delta(y, model, model_param) / 2

      probab_i_given_x = compute_probab_index(loglike_delta_x, i_flatten_x)
      probab_i_given_y = compute_probab_index(loglike_delta_y, i_flatten_y)

      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)

      return jnp.exp(loglikelihood_y - loglikelihood_x) * (
          probab_i_given_y / probab_i_given_x)

    def is_accepted(rnd_acceptance, accept_ratio):
      random_uniform_val = random.uniform(
          rnd_acceptance, shape=accept_ratio.shape, minval=0.0, maxval=1.0)
      return jnp.where(accept_ratio >= random_uniform_val, 1, 0)

    rnd_new_sample, rnd_acceptance = random.split(rnd)
    del rnd
    y, i_flatten_x, i_flatten_y = generate_new_samples(rnd_new_sample, x, model,
                                                       model_param)
    new_x = select_new_samples(rnd_acceptance, model, model_param, x, y,
                               i_flatten_x, i_flatten_y)
    new_state = state
    return new_x, new_state
