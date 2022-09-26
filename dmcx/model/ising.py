"""Ising Energy Function."""

from dmcx.model import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections
import pdb


class Ising(abstractmodel.AbstractModel):
  """Bernouli Distribution."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.dim = config.dim
    self.init_sigma = config.init_sigma
    self.lamda = config.lamda

  def make_init_params(self, rnd):

    params_weight_h = -self.lamda * jnp.ones([self.dim, self.dim - 1])
    params_weight_v = -self.lamda * jnp.ones([self.dim - 1, self.dim])
    params_b = jax.random.normal(
        rnd, shape=(self.dim, self.dim)) * self.init_sigma
    return [params_b, params_weight_h, params_weight_v]

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.choice(
        rnd,
        jnp.array([-1, 1]),
        shape=(num_samples, self.dim, self.dim),
        replace=True)
    return x0

  def forward(self, params, x):

    w_b = params[0]
    w_h = params[1]
    w_v = params[2]
    sum_neighbors = jnp.zeros([x.shape[0], self.dim, self.dim])
    sum_neighbors = sum_neighbors.at[:, :, :-1].set(
        sum_neighbors[:, :, :-1] + x[:, :, :-1] * x[:, :, 1:] * w_h)  #right
    sum_neighbors = sum_neighbors.at[:, :, 1:].set(
        sum_neighbors[:, :, 1:] + x[:, :, 1:] * x[:, :, :-1] * w_h)  # left
    sum_neighbors = sum_neighbors.at[:, :-1, :].set(
        sum_neighbors[:, :-1, :] + x[:, :-1, :] * x[:, 1:, :] * w_v)  # down
    sum_neighbors = sum_neighbors.at[:, 1:, :].set(
        sum_neighbors[:, 1:, :] + x[:, 1:, :] * x[:, :-1, :] * w_v)  #up
    biases = w_b * x
    energy_indeces = sum_neighbors + biases
    loglikelihood = jnp.sum((energy_indeces).reshape(x.shape[0], -1), axis=-1)
    return loglikelihood

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad

  # def get_expected_val(self, params):
  #   return jnp.exp(params) / (
  #       jnp.exp(params) + jnp.ones(params.shape))

  # def get_var(self, params):
  #   p = self.get_expected_val(params)
  #   return p * (1 - p)
