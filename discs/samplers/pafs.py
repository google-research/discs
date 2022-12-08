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


class PathAuxiliaryFastSampler(abstractsampler.AbstractSampler):
  """Locally Balanced Informed Sampler Class."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.adaptive = config.sampler.adaptive
    self.fixed_radius = config.sampler.fixed_radius
    self.target_acceptance_rate = config.sampler.target_acceptance_rate
    self.sample_shape = config.model.shape
    self.num_categories = config.model.num_categories
    self.balancing_fn_type = config.sampler.balancing_fn_type
    self.rank = len(self.sample_shape)

  def make_init_state(self, rnd):
    """Returns simulation time, number of log likelihood calls, and number of steps."""
    num_log_like_calls = 0
    return jnp.array([self.fixed_radius, num_log_like_calls, 0])

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
      implement different locally balanced functions in log scale
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

    rnd_radius, rnd_flip, rnd_mh = random.split(rnd, num=3)
    del rnd
    b_idx = jnp.expand_dims(jnp.arange(x.shape[0]), -1)
    expected_radius = state[0]
    radius = jnp.clip(random.poisson(rnd_radius, lam=expected_radius), a_min=1, a_max=np.product(self.sample_shape))

    if self.num_categories == 2:
      log_likelihood_x, grad_x = model.get_value_and_grad(model_param, x)
      log_likelihood_delta_x = (1. - 2. * x) * grad_x
      log_rate_x = nn.log_softmax(
          get_balancing_fn(log_likelihood_delta_x).reshape(-1, np.product(self.sample_shape)), axis=-1
      )
      # TODO(kati) line 78 has bug because of the random variable. Could you help to resolve this kati?
      _, idx_flip = lax.top_k(log_rate_x + random.gumbel(rnd_flip, shape=log_rate_x.shape), k=radius.astype(int))
      y = x.reshape(-1, np.product(self.sample_shape))
      y.at[b_idx, idx_flip].set(1. - y[b_idx, idx_flip])
      y = y.reshape(-1, *self.sample_shape)

      log_likelihood_y, grad_y = model.get_value_and_grad(model_param, y)
      log_likelihood_delta_y = (1. - 2. * y) * grad_y
      log_rate_y = nn.log_softmax(
        get_balancing_fn(log_likelihood_delta_y).reshape(-1, np.product(self.sample_shape)), axis=-1
      )

      idx_grid = (jnp.expand_dims(b_idx, -1),
                  jnp.expand_dims(jnp.arange(radius), (0, 2)),
                  jnp.expand_dims(idx_flip, 1))

      log_rate_x_expand = jnp.stack([log_rate_x for _ in range(radius)], axis=1)
      coef_x = jnp.ones_like(log_rate_x_expand)
      coef_x[idx_grid] = jnp.triu(jnp.ones((self.sample_shape[0], radius, radius)))
      log_x2y = (log_rate_x[b_idx, idx_flip].sum(-1) -
                 special.logsumexp(a=log_rate_x_expand, b=coef_x, axis=-1).sum(-1))

      log_rate_y_expand = jnp.stack([log_rate_y for _ in range(radius)], axis=1)
      coef_y = jnp.ones_like(log_rate_y_expand)
      coef_y[idx_grid] = jnp.tril(jnp.ones((self.sample_shape[0], radius, radius)))
      log_y2x = (log_rate_y[b_idx, idx_flip].sum(-1) -
                 special.logsumexp(a=log_rate_y_expand, b=coef_y, axis=-1).sum(-1))

      log_acc = log_likelihood_y + log_y2x - log_likelihood_x - log_x2y
      accepted = (random.exponential(rnd_mh, log_acc.shape) > - log_acc).astype(float).reshape(-1, *([1] * self.rank))
      new_x = y * accepted + x * (1.0 - accepted)
    else:
      raise NotImplementedError
    avg_acc = jnp.exp(jnp.clip(log_acc, a_max=0.)).mean()
    new_radius = jnp.where(self.adaptive,
        radius + (avg_acc - self.target_acceptance_rate) / (1 + state[2]) ** 0.2, radius
    )

    state.at[0].set(new_radius)
    state.at[1].set(state[1] + 4)
    state.at[2].set(state[2] + 1)
    return new_x, state


def build_sampler(config):
  return PathAuxiliaryFastSampler(config)
