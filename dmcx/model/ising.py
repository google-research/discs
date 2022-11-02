"""Ising Energy Function."""

from dmcx.model import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Ising(abstractmodel.AbstractModel):
  """Ising Distribution with Cyclic 2D Lattice."""

  def __init__(self, config: ml_collections.ConfigDict):

    self.shape = config.shape
    self.lambdaa = config.lambdaa
    self.external_field_type = config.external_field_type
    self.init_sigma = config.init_sigma
    self.num_loglike_calls = 0

  def make_init_params(self, rnd):
    # connectivity strength
    params_weight_h = self.lambdaa * jnp.ones(self.shape)
    params_weight_v = self.lambdaa * jnp.ones(self.shape)

    #TODO: Enums
    # external force (default value is zero)
    if self.external_field_type == 1:
      params_b = jax.random.normal(rnd, shape=self.shape) * self.init_sigma
      return jnp.array([params_weight_h, params_weight_v, params_b])

    return jnp.array([params_weight_h, params_weight_v])

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.randint(
        rnd,
        shape=(num_samples,) + self.shape,
        minval=0,
        maxval=2,
        dtype=jnp.int32)
    return x0

  def forward(self, params, x):

    self.num_loglike_calls += 1

    x = 2 * x - 1
    w_h = params[0][:, :-1]
    w_v = params[1][:-1, :]

    sum_neighbors = jnp.zeros((x.shape[0],) + self.shape)
    sum_neighbors = sum_neighbors.at[:, :, :-1].set(
        sum_neighbors[:, :, :-1] + x[:, :, :-1] * x[:, :, 1:] * w_h)  # right
    sum_neighbors = sum_neighbors.at[:, :, 1:].set(
        sum_neighbors[:, :, 1:] + x[:, :, 1:] * x[:, :, :-1] * w_h)  # left
    sum_neighbors = sum_neighbors.at[:, :-1, :].set(
        sum_neighbors[:, :-1, :] + x[:, :-1, :] * x[:, 1:, :] * w_v)  # down
    sum_neighbors = sum_neighbors.at[:, 1:, :].set(
        sum_neighbors[:, 1:, :] + x[:, 1:, :] * x[:, :-1, :] * w_v)  # up

    loglikelihood = sum_neighbors
    if self.external_field_type == 1:
      w_b = params[2]
      loglikelihood += w_b * x

    loglikelihood = jnp.sum((loglikelihood).reshape(x.shape[0], -1), axis=-1)
    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad

  def get_num_loglike_calls(self):
    return self.num_loglike_calls
