"""Locally Balanced Informed Sampler Class."""

import enum
from discs.samplers import abstractsampler
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


class LBWeightFn(enum.Enum):
  SQRT = 1  # sqrt(t)
  RATIO = 2  # t / (t + 1)
  MAX = 3  # max{t, 1}
  MIN = 4  # min{t, 1}


class LocallyBalancedSampler(abstractsampler.AbstractSampler):
  """Locally Balanced Informed Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):

    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.balancing_fn_type = config.sampler.balancing_fn_type

  def apply_weight_function(self, t):
    """Apply locally balanced weight function."""
    if self.balancing_fn_type == LBWeightFn.SQRT:
      return jnp.sqrt(t)
    elif self.balancing_fn_type == LBWeightFn.RATIO:
      return t / (t + 1)
    elif self.balancing_fn_type == LBWeightFn.MAX:
      return jnp.clip(t, a_min=1.0)
    elif self.balancing_fn_type == LBWeightFn.MIN:
      return jnp.clip(t, a_max=1.0)
    else:
      raise ValueError('Unknown function %s' % str(self.balancing_fn_type))

  def apply_weight_function_logscale(self, logt):
    """Apply locally balanced weight function in log scale."""
    if self.balancing_fn_type == LBWeightFn.SQRT:
      return logt / 2.0
    elif self.balancing_fn_type == LBWeightFn.RATIO:
      return jax.nn.log_sigmoid(logt)
    elif self.balancing_fn_type == LBWeightFn.MAX:
      return jnp.clip(logt, a_min=0.0)
    elif self.balancing_fn_type == LBWeightFn.MIN:
      return jnp.clip(logt, a_max=0.0)
    else:
      raise ValueError('Unknown function %s' % str(self.balancing_fn_type))

  def step(self, model, rng, x, model_param, state, x_mask=None):
    """Given the current sample, returns the next sample of the chain.

    Args:
      model: target distribution.
      rng: random key generator for JAX.
      x: current sample.
      model_param: target distribution parameters used for loglikelihood
        calulation.
      state: the state of the sampler (changes in adaptive case to tune the
        proposal distribution).

    Returns:
      New sample.
    """
    _ = x_mask
    logratio, num_calls, fn_get_neighbor = model.logratio_in_neighborhood(
        model_param, x)
    logits = self.apply_weight_function_logscale(logratio)
    sampled_idx = jax.random.categorical(rng, logits)
    new_x = fn_get_neighbor(x, sampled_idx)
    new_state = {
        'num_ll_calls': state['num_ll_calls'] + num_calls,
    }
    return new_x, new_state


def build_sampler(config):
  return LocallyBalancedSampler(config)