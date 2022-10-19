"""Gibbs with Grad Sampler Class."""

from dmcx.sampler import abstractsampler
from jax import random
import jax.numpy as jnp
import ml_collections
import math
import pdb

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
        return (-2 * x + 1) * grad  #shape of sample
      else:
        return (jnp.ones(x.shape) - x) * grad  #shape of sample with categories

    def select_index(rnd_categorical, loglikelihood):
      
      if self.num_categories == 2:
        loglikelihood_flatten = loglikelihood.reshape(loglikelihood.shape[0],
                                                      -1)
        print(jnp.shape(loglikelihood_flatten))
        return random.categorical(
            rnd_categorical, loglikelihood_flatten, axis=1)
      else:
        rnd_index, rnd_category = random.split(rnd_categorical)
        loglikelihood_over_categories = loglikelihood
        # sum over categories
        loglikelihood = jnp.sum(loglikelihood, axis=-1)
        loglikelihood_flatten = loglikelihood.reshape(loglikelihood.shape[0],
                                                      -1)
        selected_index_flatten = random.categorical(
            rnd_index, loglikelihood_flatten, axis=1)
        dim = math.prod(self.sample_shape)
        loglikelihood_over_categories_flatten = loglikelihood_over_categories.reshape(
            loglikelihood.shape[0], dim, self.num_categories)
        loglikelihood_categories = loglikelihood_over_categories_flatten[
            jnp.arange(loglikelihood.shape[0]), selected_index_flatten]
        selected_category = random.categorical(
            rnd_category, loglikelihood_categories, axis=1)
        return jnp.array([selected_index_flatten, selected_category])

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

      pdb.set_trace()

      loglike_delta_x = compute_loglike_delta(x, model, model_param) / 2
      selected_index_flatten = select_index(rnd, loglike_delta_x)

      if self.num_categories == 2:
        flipped = jnp.zeros(x.shape)
        flipped_flatten = flipped.reshape(x.shape[0], -1)
        flipped_flatten = flipped_flatten.at[jnp.arange(x.shape[0]),
                                             selected_index_flatten].set(
                                                 jnp.ones(x.shape[0]))
        flipped = flipped_flatten.reshape(x.shape)
        y = (x + flipped) % self.num_categories
      else:
        selected_index = selected_index_flatten[0]
        selected_category = selected_index_flatten[1]
        new_x = jnp.zeros([x.shape[0], self.num_categories] )
        new_x = new_x.at[jnp.arange(x.shape[0]), selected_category].set(jnp.ones(x.shape[0]))
        dim = math.prod(self.sample_shape)
        y_flatten = x.reshape(x.shape[0], dim, self.num_categories)
        y_flatten = y_flatten.at[jnp.arange(x.shape[0]), selected_index].set(new_x)
        y = y_flatten.reshape(x.shape)

      return y, selected_index_flatten

    def select_new_samples(rnd_acceptance, model, model_param, x, y, i_flatten):
      
      pdb.set_trace()
      
      accept_ratio = get_ratio(model, model_param, x, y, i_flatten)
      accepted = is_accepted(rnd_acceptance, accept_ratio)
      accepted = accepted.reshape(accepted.shape +
                                  tuple([1] * len(self.sample_shape)))
      new_x = accepted * y + (1 - accepted) * x
      return new_x

    def compute_softmax(loglikelihood):
      # pdb.set_trace()
      
      loglikelihood = (loglikelihood).reshape(loglikelihood.shape[0], -1)
      return jnp.exp(loglikelihood) / jnp.sum(
          jnp.exp(loglikelihood), axis=1, keepdims=True)

    def compute_probab_index(loglikelihood, i_flatten):

      pdb.set_trace()
      if self.num_categories == 2:
        probability = compute_softmax(loglikelihood)
        return probability[jnp.arange(loglikelihood.shape[0]), i_flatten]
      else:
        loglikelihood_category = loglikelihood[jnp.arange(loglikelihood.shape[0]), i_flatten]
        probability = compute_softmax(loglikelihood_category)
        return jnp.take(probability, i_flatten)
      
    def get_ratio(model, model_param, x, y, i_flatten):
      
      pdb.set_trace()
      

      loglike_delta_x = compute_loglike_delta(x, model, model_param)
      loglike_delta_y = compute_loglike_delta(y, model, model_param)

      probab_i_given_x = compute_probab_index(loglike_delta_x, i_flatten)
      probab_i_given_y = compute_probab_index(loglike_delta_y, i_flatten)

      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)

      return jnp.exp(loglikelihood_y - loglikelihood_x) * (
          probab_i_given_y / probab_i_given_x)

    def is_accepted(rnd_acceptance, accept_ratio):
      
      pdb.set_trace()
      
      random_uniform_val = random.uniform(
          rnd_acceptance, shape=accept_ratio.shape, minval=0.0, maxval=1.0)
      return jnp.where(accept_ratio >= random_uniform_val, 1, 0)


    rnd_new_sample, rnd_acceptance = random.split(rnd)
    del rnd
    y, i_flatten = generate_new_samples(rnd_new_sample, x, model, model_param)
    pdb.set_trace()
    new_x = select_new_samples(rnd_acceptance, model, model_param, x, y,
                               i_flatten)
    new_state = state
    return new_x, new_state
