"""Ising Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Ising(abstractmodel.AbstractModel):
  """Ising Distribution with Cyclic 2D Lattice."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape
    self.lambdaa = config.lambdaa
    self.init_sigma = config.init_sigma
    self.mu = config.mu

  def make_init_params(self, rnd):
    params = {}
    # connectivity strength
    params_weight_h = -self.lambdaa * jnp.ones(self.shape)
    params_weight_v = -self.lambdaa * jnp.ones(self.shape)

    params_b = (
        2 * jax.random.uniform(rnd, shape=self.shape) - 1
    ) * self.init_sigma
    indices = jnp.indices(self.shape)
    inner_outter = self.mu * jnp.where(
        (indices[0] / self.shape[0] - 0.5) ** 2
        + (indices[1] / self.shape[1] - 0.5) ** 2
        < 0.5 / jnp.pi,
        1,
        -1,
    )
    params_b += inner_outter
    params['params'] = jnp.array([params_weight_h, params_weight_v, params_b])
    return params

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.bernoulli(
        rnd,
        shape=(num_samples,) + self.shape,
    ).astype(jnp.int32)
    return x0

  def forward(self, params, x):
    params = params['params']
    w_h = params[0][:, :-1]
    w_v = params[1][:-1, :]
    w_b = params[2]
    message = jnp.zeros_like(x)
    message = message.at[:, :-1, :].set(
        message[:, :-1, :] + (2 * x[:, 1:, :] - 1) * w_v
    )
    message = message.at[:, 1:, :].set(
        message[:, 1:, :] + (2 * x[:, :-1, :] - 1) * w_v
    )
    message = message.at[:, :, :-1].set(
        message[:, :, :-1] + (2 * x[:, :, 1:] - 1) * w_h
    )
    message = message.at[:, :, 1:].set(
        message[:, :, 1:] + (2 * x[:, :, :-1] - 1) * w_h
    )
    message = message / 2 + w_b
    loglike = (2 * x - 1) * message
    loglike = loglike.reshape(x.shape[0], -1)
    return -jnp.sum(loglike, axis=-1)

  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad


def build_model(config):
  return Ising(config.model)
