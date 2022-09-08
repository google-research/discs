"""Bernouli Factorized Energy Function."""

from dmcx.model import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Bernouli(abstractmodel.AbstractModel):
  """Bernouli Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.dimension = config.dimension
    self.init_sigma = config.init_sigma

  def make_init_params(self, rnd):
    params = jax.random.normal(rnd, shape=(self.dimension,)) * self.init_sigma
    return params

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.randint(
        rnd,
        shape=(num_samples, self.dimension),
        minval=0,
        maxval=2,
        dtype=jnp.int32)
    return x0

  def forward(self, params, x):
    params = jnp.reshape(params, (1, -1))
    loglikelihood = jnp.sum(x * params, axis=-1)
    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad

  def get_expected_val(self, params):
    return jnp.exp(params) / (
        jnp.exp(params) + jnp.ones(params.shape[-1]))

  def get_var(self, params):
    p = self.get_expected_val(params)
    return p * (1 - p)
