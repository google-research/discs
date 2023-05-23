"""FHMM Factorized Energy Function."""

import functools
import pdb
from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class FHMM(abstractmodel.AbstractModel):
  """FHMM Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape
    self.init_sigma = config.init_sigma
    self.num_categories = config.num_categories

    self.l = config.L
    self.k = config.K
    self.sigma = config.sigma
    self.p_x0 = functools.partial(jax.random.bernoulli, p=self.alpha)
    self.p_xc = functools.partial(jax.random.bernoulli, p=1 - self.beta)

  def make_init_params(self, rnd):
    rng1, rng2, rng3, rng4 = jax.random.split(rnd, 4)
    x = self.sample_X(rng1)
    w = jax.random.normal(rng2, (self.k, 1))
    b = jax.random.normal(rng3, (1, 1))
    params = {}
    params['params'] = {}
    params['params']['Y'] = self.sample_Y(rng4, x, w, b)
    params['params']['W'] = w
    params['params']['b'] = b
    params['params']['rng'] = rng4
    return params

  def sample_X(self, rng):
    x = jnp.ones([self.l, self.k])
    x[0] = jax.random.bernoulli(rng, p=x[0] * self.alpha)
    for l in range(1, self.l):
      rng, _ = jax.random.split(rng)
      p = self.beta * x[l - 1] + (1 - self.beta) * (1 - x[l - 1])
      x[l] = jax.random.bernoulli(rng, p)
    return x

  def sample_Y(self, rng, x, w, b):
    return jax.random.normal(rng, (self.l, 1)) * self.sigma + x @ w + b

  def get_init_samples(self, rng, num_samples: int):
    x0 = jax.random.bernoulli(
        rng,
        shape=(num_samples,) + (self.l, self.k),
    )
    return x0

  # def get_init_samples(self, rnd, num_samples: int):
  #   x0 = jax.random.randint(
  #       rnd,
  #       shape=(num_samples,) + self.shape,
  #       minval=0,
  #       maxval=self.num_categories,
  #       dtype=jnp.int32,
  #   )

  #   return x0

  def forward(self, params, x):
    params = params['params']
    rng = params['rng']
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    x_0 = x[:, 0, :]
    x_cur = x[:, :-1, :]
    x_next = x[:, 1:, :]
    x_c = x_cur * (1 - x_next) + (1 - x_cur) * x_next
    logp_x = jnp.sum(jnp.log(self.p_x0(rng1, x_0)), -1) + jnp.sum(
        jnp.log(self.p_xc(rng2, x_c)), [-1, -2]
    )
    logp_y = -jnp.sum(
        jnp.square(params['Y'] - x @ params['W'] - params['b']), [-1, -2]
    ) / (2 * self.sigma**2)
    # e_x, e_y, e = self.check(x)
    loglikelihood = logp_x + logp_y
    params['rng'] = rng3
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
