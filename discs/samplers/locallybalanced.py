"""Locally Balanced Informed Sampler Class."""

import enum
from discs.common import math_util as math
from discs.common import utils
from discs.samplers import abstractsampler
import jax
import jax.numpy as jnp
import ml_collections


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

  def select_sample(
      self, rng, log_acc, current_sample, new_sample, sampler_state):
    y, acc = math.mh_step(rng, log_acc, current_sample, new_sample)
    if self.num_categories == 2:
      y = y.astype(jnp.int32)
    else:
      y = jnp.argmax(y, axis=-1)
    sampler_state = utils.copy_pytree(sampler_state)
    super().update_sampler_state(sampler_state)
    return y, sampler_state

  def apply_weight_function(self, t):
    """Apply locally balanced weight function."""
    if self.balancing_fn_type == 'SQRT':
      return jnp.sqrt(t)
    elif self.balancing_fn_type == 'RATIO':
      return t / (t + 1)
    elif self.balancing_fn_type == 'MAX':
      return jnp.clip(t, a_min=1.0)
    elif self.balancing_fn_type == 'MIN':
      return jnp.clip(t, a_max=1.0)
    else:
      raise ValueError('Unknown function %s' % str(self.balancing_fn_type))

  def apply_weight_function_logscale(self, logt):
    """Apply locally balanced weight function in log scale."""
    if self.balancing_fn_type == 'SQRT':
      return logt / 2.0
    elif self.balancing_fn_type == 'RATIO':
      return jax.nn.log_sigmoid(logt)
    elif self.balancing_fn_type == 'MAX':
      return jnp.clip(logt, a_min=0.0)
    elif self.balancing_fn_type == 'MIN':
      return jnp.clip(logt, a_max=0.0)
    else:
      raise ValueError('Unknown function %s' % str(self.balancing_fn_type))

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    if not hasattr(model, 'logratio_in_neighborhood'):
      raise ValueError('model does not have logratio_in_neighborhood function.')
    _, logratio, num_calls, fn_get_neighbor = model.logratio_in_neighborhood(
        model_param, x)
    logits = self.apply_weight_function_logscale(logratio)
    sampled_idx = jax.random.categorical(rng, logits)
    new_x = fn_get_neighbor(model_param, x, sampled_idx)
    new_state = {
        'num_ll_calls': state['num_ll_calls'] + num_calls,
    }
    return new_x, new_state


def build_sampler(config):
  return LocallyBalancedSampler(config)
