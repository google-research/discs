"""Categorical Factorized Energy Function."""

from dmcx.model import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Categorical(abstractmodel.AbstractModel):
  """Categorical Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):

    self.shape = config.shape
    self.init_sigma = config.init_sigma
    self.num_categories = config.num_categories
    self.one_hot_representation = config.one_hot_representation

  def make_init_params(self, rnd):
    params = jax.random.normal(
        rnd, shape=(self.shape + (self.num_categories,))) * self.init_sigma
    return params

  def get_init_samples(self, rnd, num_samples: int):

    x0 = jax.random.randint(
        rnd,
        shape=(num_samples,) + self.shape,
        minval=0,
        maxval=self.num_categories,
        dtype=jnp.int32)

    if self.one_hot_representation:
      return jax.nn.one_hot(x0, self.num_categories)

    return x0

  def forward(self, params, x):
    if not self.one_hot_representation:
      x = jax.nn.one_hot(x, self.num_categories)

    params = jnp.expand_dims(params, axis=0)
    loglikelihood = jnp.sum((x * params).reshape(x.shape[0], -1), axis=-1)

    return loglikelihood

  def get_value_and_grad(self, params, x):

    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)

    return loglikelihood, grad
