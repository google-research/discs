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
    if isinstance(config.sample_shape, int):
      self.sample_shape = (config.sample_shape,)
    else:
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

    def compute_delta_x(x, model, model_param):
      _, grad = model.get_value_and_grad(model_param, x)
      if self.num_categories == 2:
        return (-2 * x + 1) * grad

    def compute_index_dist(delta_x):
      delta_x_flatten = delta_x.reshape(delta_x.shape[0], -1)
      return jnp.exp(delta_x_flatten) / jnp.sum(jnp.exp(delta_x_flatten),axis=1, keepdims = True)
    
    def select_index(rnd_categorical, probability):
      return random.categorical(rnd_categorical, probability, axis=1)

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
      delta_x = compute_delta_x(x, model, model_param)
      dist_flatten = compute_index_dist(delta_x)
      index_flatten = select_index(rnd, dist_flatten)

      x_flatten = x.reshape(x.shape[0], -1)

      if self.num_categories == 2:
        new_vals = (x_flatten[jnp.arange(x.shape[0]), index_flatten] + 1) % 2
        new_x = x_flatten.at[jnp.arange(x.shape[0]),
                             index_flatten].set(new_vals)
        
        new_x = new_x.reshape((x.shape[0],) + self.sample_shape)

      return new_x

    def select_new_samples(rnd_acceptance, model, model_param, x, y):
      pdb.set_trace()
      
      accept_ratio = get_ratio(model, model_param, x, y)
      accepted = is_accepted(rnd_acceptance, accept_ratio)
      accepted = accepted.reshape(accepted.shape +
                                  tuple([1] * len(self.sample_shape)))
      new_x = accepted * y + (1 - accepted) * x
      return new_x

    def get_ratio(model, model_param, x, y):
      
      pdb.set_trace()
      
      loglikelihood_x = model.forward(model_param, x)
      loglikelihood_y = model.forward(model_param, y)
      x_dist = compute_index_dist(x)
      y_dist = compute_index_dist(y)
      return jnp.exp(loglikelihood_y - loglikelihood_x) * (y_dist / x_dist)

    def is_accepted(rnd_acceptance, accept_ratio):
      random_uniform_val = random.uniform(
          rnd_acceptance, shape=accept_ratio.shape, minval=0.0, maxval=1.0)
      return jnp.where(accept_ratio >= random_uniform_val, 1, 0)

    rnd_new_sample, rnd_acceptance = random.split(rnd)
    del rnd
    y = generate_new_samples(rnd_new_sample, x, model, model_param)
    
    new_x = select_new_samples(rnd_acceptance, model, model_param, x, y)

    new_state = state
    return new_x, new_state
