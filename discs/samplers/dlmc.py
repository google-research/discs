"""Locally Balanced Informed Sampler Class."""

from discs.samplers import abstractsampler
from discs.common.utils import log1mexp
from jax import random
from jax.scipy import special
from jax import lax
from jax import nn
import numpy as np
import jax.numpy as jnp
import ml_collections
import math


class DicreteLangevinMonteCarloSampler(abstractsampler.AbstractSampler):
  """Locally Balanced Informed Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.adaptive = config.sampler.adaptive
    self.fixed_log_tau = config.sampler.fixed_log_tau
    self.target_acceptance_rate = config.sampler.target_acceptance_rate
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.balancing_fn_type = config.sampler.balancing_fn_type
    self.rank = len(self.sample_shape)

  def make_init_state(self, rnd):
    """Returns simulation time, number of log likelihood calls, and number of steps."""
    num_log_like_calls = 0
    if self.adaptive:
      return jnp.array([-1., num_log_like_calls, 0])
    else:
      return jnp.array([self.fixed_log_tau, num_log_like_calls, 0])

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

    def get_balancing_fn(t):
      """
      implement different locally balanced function in log scale
      type_1: sqrt(t)
      type_2: t / (t + 1)
      type_3: max {t, 1}
      type_t: min {t, 1}
      """
      if self.balancing_fn_type == 2:
        return nn.log_sigmoid(t)
      elif self.balancing_fn_type == 3:  #and
        return jnp.where(t > 0., t, 0.)
      elif self.balancing_fn_type == 4:  #or
        return jnp.where(t < 0., t, 0.)
      return t / 2.

    rnd_flip, rnd_mh = random.split(rnd, num=2)
    del rnd
    log_tau = state[0]

    if self.num_categories == 2:
      log_likelihood_x, grad_x = model.get_value_and_grad(model_param, x)
      log_likelihood_delta_x = (1. - 2. * x) * grad_x
      log_rate_x = get_balancing_fn(log_likelihood_delta_x)
      log_nu_x = nn.log_sigmoid(log_likelihood_delta_x)
      log_tau = jnp.where(
        log_tau == -1, 1. - special.logsumexp(log_rate_x, axis=1+np.arange(self.rank)).mean(), log_tau)
      threshold_x = log_nu_x + log1mexp(-jnp.exp(log_tau + log_rate_x - log_nu_x))
      mask_flip = random.exponential(rnd_flip, threshold_x.shape) > - threshold_x
      y = jnp.where(mask_flip, 1. - x, x)

      log_likelihood_y, grad_y = model.get_value_and_grad(model_param, y)
      log_likelihood_delta_y = (1. - 2. * y) * grad_y
      log_rate_y = get_balancing_fn(log_likelihood_delta_y)
      log_nu_y = nn.log_sigmoid(log_likelihood_delta_y)
      threshold_y = log_nu_y + log1mexp(-jnp.exp(log_tau + log_rate_y - log_nu_y))

      mask_flip = mask_flip.astype(float)
      log_x2y = (threshold_x * mask_flip + log1mexp(threshold_x) * (1. - mask_flip)).sum(axis=1+np.arange(self.rank))
      log_y2x = (threshold_y * mask_flip + log1mexp(threshold_y) * (1. - mask_flip)).sum(axis=1+np.arange(self.rank))
      log_acc = log_likelihood_y + log_y2x - log_likelihood_x - log_x2y
      accepted = (random.exponential(rnd_mh, log_acc.shape) > - log_acc).astype(float).reshape(-1, *([1] * self.rank))
      new_x = y * accepted + x * (1.0 - accepted)
    else:
      raise NotImplementedError
    avg_acc = jnp.exp(jnp.clip(log_acc, a_max=0.)).mean()
    log_tau = jnp.where(
        self.adaptive, jnp.exp(log_tau) + (avg_acc - self.target_acceptance_rate) / (1 + state[2]) ** 0.2, log_tau
    )

    state.at[0].set(log_tau)
    state.at[1].set(state[1] + 4)
    state.at[2].set(state[2] + 1)
    return new_x, state


def build_sampler(config):
  return DicreteLangevinMonteCarloSampler(config)
