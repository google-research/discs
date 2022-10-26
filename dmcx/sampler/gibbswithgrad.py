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
        loglike_delta = (-2 * x + 1) * grad
        return loglike_delta
      else:
        # shape of sample with categories
        loglike_delta = (jnp.ones(x.shape) - x) * grad
        # loglike_delta = jnp.zeros(loglike_delta.shape)
        loglike_delta = loglike_delta - 1e9 * x
        return loglike_delta

    def select_index(rnd_categorical, loglikelihood):
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

      loglike_delta_x = compute_loglike_delta(x, model, model_param)/2
      selected_index_flatten = select_index(rnd, loglike_delta_x)
      selected_index_flatten_y = selected_index_flatten

      if self.num_categories == 2:
        flipped = jnp.zeros(x.shape)
        flipped_flatten = flipped.reshape(x.shape[0], -1)
        flipped_flatten = flipped_flatten.at[jnp.arange(x.shape[0]),
                                             selected_index_flatten].set(
                                                 jnp.ones(x.shape[0]))
        flipped = flipped_flatten.reshape(x.shape)
        y = (x + flipped) % self.num_categories
      else:
        dim = math.prod(self.sample_shape)
        selected_index = jnp.floor_divide(selected_index_flatten,
                                          self.num_categories)
        selected_category = selected_index_flatten % self.num_categories
        new_x = jnp.zeros([x.shape[0], self.num_categories])
        new_x = new_x.at[jnp.arange(x.shape[0]),
                         selected_category].set(jnp.ones(x.shape[0]))
        y_flatten = x.reshape(x.shape[0], dim, self.num_categories)
        y_flatten = y_flatten.at[jnp.arange(x.shape[0]),
                                 selected_index].set(new_x)
        y = y_flatten.reshape(x.shape)
        selected_index_flatten_y = jnp.where(
            (y - x).reshape(x.shape[0], -1) == -1)[1]
  
      return y, selected_index_flatten, selected_index_flatten_y

    def select_new_samples(rnd_acceptance, model, model_param, x, y, i_flatten_x, i_flatten_y):

      accept_ratio = get_ratio(model, model_param, x, y, i_flatten_x,
                               i_flatten_y)
      # print(accept_ratio)
      accepted = is_accepted(rnd_acceptance, accept_ratio)
      # print(accepted)
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


# [[1 0 0][0 1 0]]
# pdb.set_trace()
# score_change_x = grad - (grad * x).sum(dim=-1, keepdim=True)
# score_change_x = score_change_x - 1e9 * x
# dist_x = StableOnehotCategorical(logits=score_change_x.view(bsize, -1))
# index_x = dist_x.sample()
# log_x2y = dist_x.log_prob(index_x)
# index_y = x * index_x.view(x.shape).sum(dim=-1, keepdim=True)
# y = x * (1 - index_x.view(x.shape).sum(dim=-1, keepdim=True)) + index_x.view(x.shape)
# return x
# else:
#   dim = math.prod(self.sample_shape)
#   selected_index = (i_flatten / dim).astype(jnp.int32)
#   selected_category = i_flatten % dim
#   loglikelihood = loglikelihood.reshape(loglikelihood.shape[0], dim,
#                                         self.num_categories)
#   loglikelihood_category = loglikelihood[
#       jnp.arange(loglikelihood.shape[0]), selected_index]
#   probability = compute_softmax(loglikelihood_category)
#   return probability[jnp.arange(loglikelihood.shape[0]),
#                      selected_category]
