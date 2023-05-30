"""Bernoulli Factorized Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Bernoulli(abstractmodel.AbstractModel):
  """Bernoulli Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape
    self.init_sigma = config.init_sigma

  def make_init_params(self, rng):
    params = {}
    params['params'] = (
        jax.random.normal(rng, shape=self.shape) * self.init_sigma
    )
    return params

  def get_init_samples(self, rng, num_samples: int):
    x0 = jax.random.randint(
        rng,
        shape=(num_samples,) + self.shape,
        minval=0,
        maxval=2,
        dtype=jnp.int32,
    )
    return x0

  def forward(self, params, x):
    params = params['params']
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

  def get_neighbor_fn(self, x, neighbhor_idx):
    x_shape = x.shape
    x = jnp.reshape(x, (x.shape[0], -1))
    brange = jnp.arange(x.shape[0])
    cur_val = x[brange, neighbhor_idx]
    y = x.at[brange, neighbhor_idx].set(1 - cur_val)
    return jnp.reshape(y, x_shape)

  def logratio_in_neighborhood(self, params, x):
    params = params['params']
    params = jnp.expand_dims(params, axis=0)
    diff = (1 - 2 * x) * params
    return diff, 1, self.get_neighbor_fn

  def get_expected_val(self, params):
    params = params['params']
    return jnp.exp(params) / (jnp.exp(params) + jnp.ones(params.shape))

  def get_var(self, params):
    p = self.get_expected_val(params)
    return p * (1 - p)


def build_model(config):
  return Bernoulli(config.model)
