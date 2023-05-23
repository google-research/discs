"""FHMM Factorized Energy Function."""

import functools
import pdb
from discs.common import math_util as math
from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class FHMM(abstractmodel.AbstractModel):
  """FHMM Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape
    self.num_categories = config.num_categories
    self.l = self.shape[0]
    self.k = self.shape[1]
    self.sigma = config.sigma
    self.alpha = config.alpha
    self.beta = config.beta

  def make_init_params(self, rnd):
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rnd, 5)
    x = self.sample_X(rng1)
    w = jax.random.normal(rng2, (self.k, 1))
    b = jax.random.normal(rng3, (1, 1))
    y = self.sample_Y(rng4, x, w, b)
    params = {}
    params['params'] = jnp.concatenate((w, y, b), axis=0)
    return params

  def sample_X(self, rng):
    x = jnp.ones([self.l, self.k])
    x = x.at[0].set(jax.random.bernoulli(rng, p=x[0] * self.alpha))
    for l in range(1, self.l):
      rng, _ = jax.random.split(rng)
      p = self.beta * x[l - 1] + (1 - self.beta) * (1 - x[l - 1])
      x = x.at[l].set(jax.random.bernoulli(rng, p))
    return x

  def sample_Y(self, rng, x, w, b):
    return (
        jax.random.normal(rng, (self.l, 1)) * self.sigma +jnp.matmul(x, w) + b
    )

  def log_probab_of_px(self, x, p):
    num_ones = jnp.sum(x)
    dim = math.prod(x.shape)
    #prob = (p ** (num_ones)) * ((1 - p) ** (dim - num_ones))
    prob = p*x + (1-p)*(1-x) 
    return jnp.log(prob)

  def get_init_samples(self, rng, num_samples: int):
    x0 = jax.random.bernoulli(
        rng,
        shape=(num_samples,) + (self.l, self.k),
    )
    return x0

  def forward(self, params, x):
    params = params['params']
    w = params[0:self.k, :]
    y = params[self.k:self.k+self.l, :]
    b = params[-1:, :]
    logp_y = -jnp.sum(
        jnp.square(y - jnp.matmul(x, w) - b), [-1, -2]
    ) / (2 * self.sigma**2)

    x_0 = x[:, 0, :]
    x_cur = x[:, :-1, :]
    x_next = x[:, 1:, :]
    x_c = x_cur * (1 - x_next) + (1 - x_cur) * x_next
    logp_x = jnp.sum(self.log_probab_of_px(x_0, self.alpha), -1) + jnp.sum(
        jnp.log(self.log_probab_of_px(x_c, 1 - self.beta)), [-1, -2]
    )
    #logp_y = -jnp.sum(
    #    jnp.square(self.y - jnp.matmul(x, w) - self.b), [-1, -2]
    #) / (2 * self.sigma**2)
    # e_x, e_y, e = self.check(x)
    loglikelihood = logp_x + logp_y
    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)

    return loglikelihood, grad


def build_model(config):
  return FHMM(config.model)
