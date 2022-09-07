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
    energy = jnp.sum(x * params, axis=-1)
    return energy

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      energy = self.forward(params, z)
      return jnp.sum(energy), energy

    (_, energy), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return energy, grad

  def get_expected_val(self, params):
    # for clarity x0 and x1 are defined!
    x1 = jnp.ones(params.shape[-1])
    x0 = jnp.zeros(params.shape[-1])
    return jnp.exp(-params * x1) / (
        jnp.exp(-params * x1) + jnp.exp(-params * x0))

  def get_var(self, params):
    p = self.get_expected_val(params)
    return p * (1 - p)
